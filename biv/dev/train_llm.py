import click
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from bitsandbytes.optim import AdamW8bit
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor
from torch.utils.data import Dataset
from torchmetrics import MeanMetric, Metric
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from xztrainer import XZTrainable, BaseContext, DataType, ModelOutputsType, ContextType, set_seeds, enable_tf32, \
    XZTrainer, XZTrainerConfig

from biv.dev.common_train import stack_pad
from biv.dev.dto import Payment
from biv.dev.prompting import make_prompt_train, BASE_MODEL_NAME


def prepare_model_for_lora(device):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=64, lora_dropout=0.1,
        use_rslora=True,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']  # att
    )
    model = get_peft_model(model, peft_config)
    return model


def find_sub_list(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            return ind


class PayDataset(Dataset):
    def __init__(self):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        self._dataset = pd.read_csv('data/for-teams/payments_training.tsv', sep='\t',
                                    names=['id', 'date', 'sum', 'description', 'category'])

    def __getitem__(self, item_n):
        item = self._dataset.iloc[item_n]
        item = Payment(id=item['id'], date=item['date'], sum=item['sum'], description=item['description'],
                       category=item['category'])
        prompt = make_prompt_train(item)
        template = self._tokenizer.apply_chat_template(prompt)
        assistant_starter = [128006, 78191, 128007]
        assistant_starter_idx = find_sub_list(assistant_starter, template)
        input_ids = torch.tensor(template, dtype=torch.long)
        labels = input_ids.clone()
        labels[: assistant_starter_idx] = -100
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones((len(template),), dtype=torch.long)
        }

    def __len__(self):
        return len(self._dataset)


class PayCollator:
    def __call__(self, batch):
        return {
            'input_ids': stack_pad([x['input_ids'] for x in batch], 0),
            'labels': stack_pad([x['input_ids'] for x in batch], -100),
            'attention_mask': stack_pad([x['attention_mask'] for x in batch], 0),
        }


class PayTrainable(XZTrainable):
    def step(self, context: BaseContext, data: DataType) -> tuple[Tensor, ModelOutputsType]:
        outs = context.model(**data)
        return outs.loss, {
            'loss': outs.loss
        }

    def create_metrics(self, context_type: ContextType) -> dict[str, Metric]:
        return {
            'loss': MeanMetric()
        }

    def update_metrics(self, context_type: ContextType, model_outputs: dict[str, list], metrics: dict[str, Metric]):
        metrics['loss'].update(model_outputs['loss'])


def train():
    accel = Accelerator(
        gradient_accumulation_steps=4,
        log_with='tensorboard',
        project_dir='.'
    )
    set_seeds(0xFAFA)
    enable_tf32()
    with torch.no_grad():
        lora_device = torch.empty(1).to(accel.device).device
    model = prepare_model_for_lora(lora_device)

    trainer = XZTrainer(
        config=XZTrainerConfig(
            experiment_name='train-vikhr',
            minibatch_size=1,
            minibatch_size_eval=1,
            epochs=1,
            gradient_clipping=5.0,
            optimizer=lambda module: AdamW8bit(module.parameters(), lr=5e-5, weight_decay=1e-4),
            scheduler=lambda optimizer, total_steps: get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.05),
                                                                                     total_steps),
            collate_fn=PayCollator(),
            dataloader_persistent_workers=False,
            dataloader_num_workers=4,
            dataloader_shuffle_train_dataset=True,
            dataloader_pin_memory=True,
            tracker_logging_dir='./logs',
            log_steps=10,
            save_steps=1000,
            eval_steps=1000
        ),
        model=model,
        trainable=PayTrainable(),
        accelerator=accel
    )
    trainer.train(PayDataset(), None)
    model.save_pretrained('checkpoint/train-vikhr')


if __name__ == '__main__':
    train()
