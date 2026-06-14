from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DEFAULT_GENERATION_MODEL = "CohereLabs/tiny-aya-water"


def load_generation_pipeline(model_name: str = DEFAULT_GENERATION_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

