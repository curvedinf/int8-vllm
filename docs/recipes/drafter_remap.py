#!/usr/bin/env python3
"""Post-bake remap for the DFlash2 int8 drafter checkpoint.

gptqmodel saves the drafter with three defects vs what vLLM's loader expects:
1. weights prefixed 'model.' (vLLM's DFlash2Qwen3Model is itself the model)
2. architectures tag rewritten to 'Qwen3ForCausalLM' (breaks DFlash2 detect)
3. conv/selector/norm params quantized (they must stay dense)

Usage: python drafter_remap.py            # remap in place
       python drafter_remap.py --dequant  # also dequantize Linears to dense fp16
"""
import argparse, json, os, sys
import torch
from safetensors.torch import load_file, save_file

QDIR = os.path.expanduser(
    "~/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit")

SKIP_DENSE = ('conv', 'candidate_selector', 'fc.', 'lm_head', 'norm.',
              'q_norm', 'k_norm', 'input_layernorm', 'post_attention_layernorm',
              'embed_tokens')

def remap(dequant: bool):
    t = load_file(f'{QDIR}/model.safetensors') if os.path.exists(f'{QDIR}/model.safetensors') \
        else merge_shards()
    out = {}
    for k, v in t.items():
        base = k[len('model.'):] if k.startswith('model.') else k
        if any(s in k for s in SKIP_DENSE):
            continue
        out[base] = v
    # restore dense params from bf16 source (skip modules that carry GPTQ
    # tensors — restoring their dense .weight leaves the loader two sources)
    bf16 = load_file(os.path.expanduser(
        '~/models/dflash2-bf16-with-tokenizer/model.safetensors'))
    quantized = {k.rsplit('.', 1)[0] for k in out if k.endswith('.qweight')}
    for k, v in bf16.items():
        if k.endswith('.weight') and k[:-len('.weight')] in quantized:
            continue
        out.setdefault(k, v)
    if dequant:
        prefixes = {k.rsplit('.', 1)[0] for k in out if k.endswith('.qweight')}
        for p in list(prefixes):
            qw = out[f'{p}.qweight'].to(torch.int32)  # [K//4, N]
            sc = out[f'{p}.scales'].float()           # [K//G, N]
            K4, N = qw.shape; K = K4 * 4
            sh = torch.arange(4, device=qw.device, dtype=torch.int32) * 8
            w = ((qw.unsqueeze(-1) >> sh) & 0xFF).reshape(K, N).to(torch.int16) - 128
            G = K // sc.shape[0]
            out[f'{p}.weight'] = (w.float().reshape(K//G, G, N) * sc.unsqueeze(1)
                                  ).reshape(K, N).to(torch.float16)
            for s in ('.qweight', '.qzeros', '.scales', '.g_idx'):
                out.pop(f'{p}{s}', None)
    save_file(out, f'{QDIR}/model.safetensors')
    cpath = f'{QDIR}/config.json'
    c = json.load(open(cpath))
    c['architectures'] = ['DFlash2DraftModel']
    if dequant:
        c.pop('quantization_config', None)
    json.dump(c, open(cpath, 'w'), indent=2)
    idx = {'metadata': {'format': 'pt'},
           'weight_map': {k: 'model.safetensors' for k in out}}
    json.dump(idx, open(f'{QDIR}/model.safetensors.index.json', 'w'), indent=2)
    print(f'remapped {len(out)} tensors (dequant={dequant})')

def merge_shards():
    idx = json.load(open(f'{QDIR}/model.safetensors.index.json'))
    shards = {}
    for k, s in idx['weight_map'].items():
        shards.setdefault(s, {})[k] = True
    t = {}
    for s in shards:
        t.update(load_file(f'{QDIR}/{s}'))
        os.remove(f'{QDIR}/{s}')
    return t

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dequant', action='store_true')
    remap(ap.parse_args().dequant)
