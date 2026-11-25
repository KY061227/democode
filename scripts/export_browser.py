"""Merge the group's LoRA adapter and export a browser inference model."""
import json
import os
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from peft import PeftModel
from spellchecker import SpellChecker
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'web' / 'model'
OUT.mkdir(parents=True, exist_ok=True)
ADAPTER = ROOT / 'models' / 'opt-125m-lora-adapter'
BASE = 'facebook/opt-125m'

class BrowserOPT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, return_dict=False)[0][:, -1, :]

def main():
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER))
    tokenizer.save_pretrained(str(OUT))
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32,
                                              attn_implementation='eager')
    model = PeftModel.from_pretrained(base, str(ADAPTER)).merge_and_unload().float().cpu().eval()
    wrapper = BrowserOPT(model).eval()
    ids = torch.tensor([[2, 100, 200, 300]], dtype=torch.long)
    mask = torch.ones_like(ids)
    fp32 = OUT / 'model-fp32.onnx'
    with torch.no_grad():
        torch.onnx.export(wrapper, (ids, mask), str(fp32), input_names=['input_ids','attention_mask'],
                          output_names=['logits'], opset_version=17, dynamo=False,
                          dynamic_axes={'input_ids':{0:'batch',1:'sequence'},
                                        'attention_mask':{0:'batch',1:'sequence'},
                                        'logits':{0:'batch'}}, external_data=False)
    target = OUT / 'model.onnx'
    quantize_dynamic(str(fp32), str(target), weight_type=QuantType.QUInt8,
                     op_types_to_quantize=['MatMul'], per_channel=False)
    # Smoke-test variable lengths before publishing any build.
    session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    for text in ['Do you', 'My phone battery is at 1%, do you have a']:
        tokens = tokenizer(text, return_tensors='np')
        result = session.run(None, {'input_ids':tokens['input_ids'].astype(np.int64),
                                   'attention_mask':tokens['attention_mask'].astype(np.int64)})[0]
        assert result.ndim == 2 and np.isfinite(result).all()
    fp32.unlink()
    spell = SpellChecker(distance=1)
    dictionary = {w: n for w,n in spell.word_frequency.dictionary.items()
                  if w.isascii() and w.isalpha() and len(w)<=24}
    (OUT/'dictionary.json').write_text(json.dumps(dictionary), encoding='utf-8')
    manifest = {'model':'OPT-125M + your LoRA adapter · ONNX INT8',
                'bytes':target.stat().st_size,'bos':tokenizer.bos_token_id,
                'pad':tokenizer.pad_token_id,'contextLength':128}
    (OUT/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    print('Browser export validated:', manifest, flush=True)

if __name__ == '__main__':
    main()
