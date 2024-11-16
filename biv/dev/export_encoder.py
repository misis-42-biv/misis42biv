import json

import torch
from safetensors.torch import load_file
from tokenizers.tokenizers import Tokenizer
from transformers import AutoTokenizer

from biv.dev.train_encoder import EncoderModel, CLASS_MAP, BASE_ENCODER_MODEL


def export():
    checkpoint = load_file('checkpoint/train-encoder/model.safetensors')
    model = EncoderModel().eval()
    model.load_state_dict(checkpoint)
    torch.onnx.export(
        model,
        (torch.tensor([[1, 2, 3]], dtype=torch.long), torch.tensor([[1, 1, 1]], dtype=torch.long)),
        "export/model/encoder.onnx",
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'attention_mask': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'class_count'}
        }
    )
    with open('export/model/encoder.json', 'w') as f:
        json.dump(CLASS_MAP, f)
    tokenizer = AutoTokenizer.from_pretrained(BASE_ENCODER_MODEL)
    tokenizer.save_pretrained('export/model/tokenizer')


if __name__ == '__main__':
    export()
