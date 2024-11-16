from pathlib import Path

import click
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from biv.dev.dto import Payment
from biv.dev.prompting import make_prompt, BASE_MODEL_NAME


@click.command()
@click.option('--device', type=str, default='cuda:0')
def label(device: str):
    df = pd.read_csv('data/for-teams/payments_main.tsv', sep='\t',
                                    names=['id', 'date', 'sum', 'description'])
    results = []

    if Path('data/autolabel_encoder.parquet').is_file():
        already = pd.read_parquet('data/autolabel_encoder.parquet')
        results.extend(already['category'])


    payments = [Payment(id=x.id, date=x.date, sum=x.sum, description=x.description, category=None) for x in df.itertuples()][len(results):]
    prompts = [make_prompt(pay) for pay in payments]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained('checkpoint/train-vikhr', torch_dtype=torch.bfloat16, device_map=device).eval()

    for i, (prompt, payment) in tqdm(enumerate(zip(prompts, payments)), total=len(prompts)):
        template = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        template = torch.tensor(template, dtype=torch.long, device=device)[None, :]
        result = model.generate(template, temperature=0.0, do_sample=False, top_p=None, max_new_tokens=15, num_beams=4, use_cache=True)
        result_text = tokenizer.decode(result[0, template.shape[1]:-1])
        results.append(result_text)

        if i % 50 == 0:
            minidf = df.iloc[:len(results)].copy()
            minidf['category'] = results
            minidf.to_parquet('data/autolabel_encoder.parquet')



if __name__ == '__main__':
    label()
