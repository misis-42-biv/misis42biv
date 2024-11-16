import json

import numpy as np
import pandas as pd
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from tokenizers import Tokenizer
from tqdm import tqdm


def _load_data() -> pd.DataFrame:
    data = pd.read_csv('/data/input.tsv', sep='\t', names=['id', 'date', 'sum', 'description'])
    data['model_input'] = data.apply(lambda x: f'Дата: {x["date"]}\nСумма: {x["sum"]}\nОписание: {x["description"]}',
                                     axis=1)
    data['model_input_len'] = data['model_input'].str.len()
    data = data.sort_values(by='model_input_len')
    return data


def _load_model() -> tuple[Tokenizer, InferenceSession]:
    tokenizer = Tokenizer.from_file('/model/tokenizer/tokenizer.json')

    model_opts = SessionOptions()
    model_opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    model = InferenceSession('/model/encoder.onnx', model_opts)

    return tokenizer, model


def _load_class_map() -> dict[int, str]:
    with open('/model/encoder.json', 'r') as f:
        class_map = json.load(f)
    class_map = {v: k for k, v in class_map.items()}
    return class_map


_BATCH_SIZE = 8


def _predict_loop(data: pd.DataFrame, tokenizer: Tokenizer, model: InferenceSession, class_map: dict[int, str]) -> list[dict]:
    results = []
    for batch_start in tqdm(range(0, len(data), _BATCH_SIZE)):
        batch_data = data.iloc[batch_start:batch_start + _BATCH_SIZE]
        input_enc = tokenizer.encode_batch(list(batch_data['model_input']))
        logits, = model.run(['logits'], {
            'input_ids': np.asarray([x.ids for x in input_enc]),
            'attention_mask': np.asarray([x.attention_mask for x in input_enc])
        })
        class_ids = np.argmax(logits, axis=-1)
        class_names = [class_map[x.item()] for x in class_ids]

        for idx, class_name in zip(batch_data['id'], class_names):
            results.append({'id': idx, 'category': class_name})
    return results


def _save_results(results: list[dict]):
    df = pd.DataFrame(results)
    df = df.sort_values(by='id')
    df.to_csv('/data/output.tsv', header=False, index=False, sep='\t')


def main():
    data = _load_data()
    tokenizer, model = _load_model()
    class_map = _load_class_map()
    results = _predict_loop(data, tokenizer, model, class_map)
    _save_results(results)



if __name__ == '__main__':
    main()