"""Load real Llama weights with Mini-SGLang; construct a request and a batch.

Run in a fresh process: source ../../activate-minisgl.sh; python minimal_llama.py
This example does not allocate a serving KV pool or run model inference.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from minisgl.core import Batch, Context, Req, SamplingParams, set_global_ctx
from minisgl.distributed import set_tp_info
from minisgl.kvcache.naive_cache import NaivePrefixCache
from minisgl.layers import set_rope_device
from minisgl.models import BaseLLMModel, ModelConfig, create_model, load_weight
from minisgl.utils import torch_dtype


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    local_model = Path(__file__).resolve().parents[2] / 'models' / 'TinyLlama-1.1B-Chat-v1.0'
    default_model = str(local_model) if local_model.is_dir() else 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    parser.add_argument('--model', default=default_model)
    parser.add_argument('--prompt', default='Hello, my name is')
    args = parser.parse_args()
    device = torch.device('cuda:0')
    dtype = torch.float16
    if not torch.cuda.is_available():
        raise RuntimeError('Activate the Mini-SGLang environment on a CUDA GPU machine.')

    print('[1] Load configuration', flush=True)
    hf_config = AutoConfig.from_pretrained(args.model)
    if hf_config.model_type != 'llama':
        raise ValueError(f'Expected a Llama model, got {hf_config.model_type}')
    config = ModelConfig.from_hf(hf_config)
    print('Model:', args.model)
    print('HF config:', type(hf_config).__name__)
    print(json.dumps(asdict(config), indent=2))

    # Mini-SGLang uses process-global TP information, even for a single GPU.
    set_tp_info(rank=0, size=1)
    set_rope_device(device)
    # Meta tensors allocate no weight storage; RoPE cache uses the real device.
    with torch.device('meta'), torch_dtype(dtype):
        model = create_model(config)
    # The loader performs checkpoint name conversion, TP slicing and QKV merging.
    weights = {name: value.to(dtype=dtype) for name, value in load_weight(args.model, device)}
    tensor_count = len(weights)
    model.load_state_dict(weights)  # Consumes the dict and checks shape/dtype.
    state = model.state_dict()
    assert isinstance(model, BaseLLMModel)
    assert state and all(t.device == device and t.dtype == dtype for t in state.values())
    assert not weights
    print('\n[2] Pretrained weights loaded', flush=True)
    print('Model class:', type(model).__name__)
    print('isinstance(model, BaseLLMModel):', isinstance(model, BaseLLMModel))
    print('Weight tensors:', tensor_count)
    print('Parameter elements:', sum(t.numel() for t in state.values()))
    print('Device / dtype:', device, dtype)
    print('QKV weight:', tuple(model.model.layers.op_list[0].self_attn.qkv_proj.weight.shape))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer.encode(args.prompt, return_tensors='pt')[0].to(torch.int32)
    params = SamplingParams(temperature=0.0, max_tokens=8)
    # Obtain a real empty cache handle, instead of supplying None.
    prefix_cache = NaivePrefixCache(device)
    handle = prefix_cache.match_prefix(input_ids).cuda_handle
    req = Req(input_ids=input_ids, table_idx=0, cached_len=handle.cached_len,
              output_len=params.max_tokens, uid=1, sampling_params=params, cache_handle=handle)
    batch = Batch(reqs=[req], phase='prefill')
    # Manually illustrate the token tensors normally supplied by Scheduler.
    batch.padded_reqs = list(batch.reqs)
    batch.input_ids = req.input_ids[req.cached_len:req.device_len].to(device)
    batch.positions = torch.arange(req.cached_len, req.device_len, dtype=torch.int32, device=device)
    print('\n[3] Manually constructed Req and Batch')
    print('Prompt:', repr(args.prompt))
    print('CPU token ids:', req.input_ids.tolist())
    print(f'Req: uid={req.uid}, table_idx={req.table_idx}, cached_len={req.cached_len}, '
          f'device_len={req.device_len}, max_device_len={req.max_device_len}')
    print(f'Req: extend_len={req.extend_len}, remain_len={req.remain_len}, can_decode={req.can_decode}')
    print(f'Batch: phase={batch.phase}, size={batch.size}, padded_size={batch.padded_size}')
    print('Batch.input_ids:', tuple(batch.input_ids.shape), batch.input_ids.device)
    print('Batch.positions:', batch.positions.tolist())
    assert batch.size == batch.padded_size == 1
    assert len(batch.input_ids) == req.extend_len

    ctx = Context(page_size=1)
    set_global_ctx(ctx)
    with ctx.forward_batch(batch):
        assert ctx.batch is batch
        print('\n[4] Context: active batch is our batch ->', ctx.batch is batch)
    assert ctx._batch is None
    print('Context: active batch cleared after scope ->', ctx._batch is None)
    print('No forward pass: KV pool, page table, out_loc and attention metadata are not initialized.')
    print('PASS: real Llama weights loaded; configuration, Req, Batch and Context verified.', flush=True)


if __name__ == '__main__':
    main()
