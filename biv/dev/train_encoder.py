import pandas as pd
import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torchmetrics import Metric, MeanMetric, Accuracy
from transformers import AutoTokenizer, AutoModel, AdamW, get_cosine_schedule_with_warmup
from xztrainer import set_seeds, enable_tf32, XZTrainable, BaseContext, DataType, ModelOutputsType, ContextType, \
    XZTrainer, XZTrainerConfig

from biv.dev.common_train import stack_pad
from biv.dev.dto import Payment

BASE_ENCODER_MODEL = 'Tochka-AI/ruRoPEBert-e5-base-2k'

CLASS_MAP = {
    'BANK_SERVICE': 0,
    'FOOD_GOODS': 1,
    'LEASING': 2,
    'LOAN': 3,
    'NON_FOOD_GOODS': 4,
    'NOT_CLASSIFIED': 5,
    'REALE_STATE': 6,
    'SERVICE': 7,
    'TAX': 8
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_orig = pd.read_csv('data/for-teams/payments_training.tsv', sep='\t',
                          names=['id', 'date', 'sum', 'description', 'category'])
    df_orig_train, df_orig_test = train_test_split(df_orig, test_size=0.4, stratify=df_orig['category'])
    df_new = pd.read_parquet('data/autolabel_encoder.parquet')
    return pd.concat((df_orig_train, df_new)), df_orig_test


class EncoderPayDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(BASE_ENCODER_MODEL)
        self._dataset = dataset

    def __getitem__(self, item_n):
        item = self._dataset.iloc[item_n]
        item = Payment(id=item['id'], date=item['date'], sum=item['sum'], description=item['description'],
                       category=item['category'])
        item_inputs = self._tokenizer.encode_plus(f'Дата: {item.date}\nСумма: {item.sum}\nОписание: {item.description}').encodings[0]
        return {
            'input_ids': torch.tensor(item_inputs.ids, dtype=torch.long),
            'attention_mask': torch.tensor(item_inputs.attention_mask, dtype=torch.long),
            'target': torch.scalar_tensor(CLASS_MAP[item.category], dtype=torch.long)
        }

    def __len__(self):
        return len(self._dataset)


class EncoderPayCollator:
    def __call__(self, batch):
        return {
            'input_ids': stack_pad([x['input_ids'] for x in batch], 0),
            'attention_mask': stack_pad([x['attention_mask'] for x in batch], 0),
            'target': torch.stack([x['target'] for x in batch])
        }


class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = AutoModel.from_pretrained(BASE_ENCODER_MODEL)
        self._cls = nn.Linear(self._model.config.hidden_size, len(CLASS_MAP))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self._model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self._cls(pooled)


class EncoderTrainable(XZTrainable):
    def __init__(self):
        super().__init__()
        self._loss = CrossEntropyLoss()

    def step(self, context: BaseContext, data: DataType) -> tuple[Tensor, ModelOutputsType]:
        outs = context.model(data['input_ids'], data['attention_mask'])
        loss = self._loss(outs, data['target'])
        return loss, {
            'loss': loss,
            'predict_proba': torch.softmax(outs, dim=-1),
            'target': data['target']
        }

    def create_metrics(self, context_type: ContextType) -> dict[str, Metric]:
        return {
            'loss': MeanMetric(),
            'accuracy': Accuracy(task='multiclass', num_classes=len(CLASS_MAP))
        }

    def update_metrics(self, context_type: ContextType, model_outputs: dict[str, list], metrics: dict[str, Metric]):
        metrics['loss'].update(model_outputs['loss'])
        metrics['accuracy'].update(model_outputs['predict_proba'], model_outputs['target'])


def train():
    df_train, df_test = load_data()
    df_train = df_train[df_train['category'].isin(CLASS_MAP.keys())]
    ds_train = EncoderPayDataset(df_train)
    ds_test = EncoderPayDataset(df_test)

    accel = Accelerator(
        gradient_accumulation_steps=4,
        log_with='tensorboard',
        project_dir='.',
    )
    set_seeds(0xFAFA)
    enable_tf32()
    model = EncoderModel()

    trainer = XZTrainer(
        config=XZTrainerConfig(
            experiment_name='train-encoder',
            minibatch_size=4,
            minibatch_size_eval=1,
            epochs=1,
            gradient_clipping=5.0,
            optimizer=lambda module: AdamW(module.parameters(), lr=5e-5, weight_decay=1e-4),
            scheduler=lambda optimizer, total_steps: get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1),
                                                                                     total_steps),
            collate_fn=EncoderPayCollator(),
            dataloader_persistent_workers=False,
            dataloader_num_workers=4,
            dataloader_shuffle_train_dataset=True,
            dataloader_pin_memory=True,
            tracker_logging_dir='./logs',
            log_steps=10,
            save_steps=500,
            eval_steps=500
        ),
        model=model,
        trainable=EncoderTrainable(),
        accelerator=accel
    )
    trainer.train(ds_train, ds_test)
    save_file(model.state_dict(), 'checkpoint/train-encoder/model.safetensors')


if __name__ == '__main__':
    train()
