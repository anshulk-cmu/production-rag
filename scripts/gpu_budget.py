"""Measure local-profile VRAM for bge-m3 + bge-reranker-v2-m3 + Llama-3.2-3B (bf16/fp16)
and decide what must go on CPU RAM to fit. Run: python scripts/gpu_budget.py
"""

import gc
import time

import torch
from huggingface_hub import login

from rag.config import get_settings

EMB_ID = "BAAI/bge-m3"
RR_ID = "BAAI/bge-reranker-v2-m3"
LLM_ID = "meta-llama/Llama-3.2-3B-Instruct"
ST_DTYPE = torch.float16
LLM_DTYPE = torch.bfloat16

s = get_settings()
if s.hf_token:
    login(token=s.hf_token, add_to_git_credential=False)

TOTAL = torch.cuda.get_device_properties(0).total_memory / 1024**3


def reserved() -> float:
    return torch.cuda.memory_reserved() / 1024**3


def load_embedder(device):
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(EMB_ID, device=device, model_kwargs={"dtype": ST_DTYPE})
    m.encode(["warmup"], normalize_embeddings=True)
    return m


def load_reranker(device):
    from sentence_transformers import CrossEncoder

    m = CrossEncoder(RR_ID, device=device, model_kwargs={"dtype": ST_DTYPE})
    m.predict([("q", "warmup passage")])
    return m


def load_llm(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLM_ID)
    model = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=LLM_DTYPE, device_map={"": device})
    return model, tok


def smoke(llm, tok) -> float:
    msgs = [{"role": "user", "content": "Name three primary colors."}]
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(llm.device)
    llm.generate(**inputs, max_new_tokens=32, do_sample=False)
    return torch.cuda.max_memory_allocated() / 1024**3


def trial(reranker_device: str) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"\n=== Trial: reranker on {reranker_device.upper()} (embedder+LLM on GPU) ===")
    r0 = reserved()
    t = time.time()
    emb = load_embedder("cuda")
    r1 = reserved()
    print(
        f"  embedder bge-m3 fp16 [{emb.device}] {time.time()-t:.0f}s | +{r1-r0:.2f} GB | reserved {r1:.2f}"
    )
    t = time.time()
    rr = load_reranker(reranker_device)
    r2 = reserved()
    rr_dev = next(rr.model.parameters()).device
    print(
        f"  reranker bge-reranker-v2-m3 fp16 [{rr_dev}] {time.time()-t:.0f}s | +{r2-r1:.2f} GB | reserved {r2:.2f}"
    )
    t = time.time()
    llm, tok = load_llm("cuda")
    r3 = reserved()
    print(
        f"  LLM Llama-3.2-3B bf16 [{next(llm.parameters()).device}] {time.time()-t:.0f}s | +{r3-r2:.2f} GB | reserved {r3:.2f}"
    )
    peak = smoke(llm, tok)
    headroom = TOTAL - reserved()
    print(
        f"  after generate: reserved {reserved():.2f} GB | peak alloc {peak:.2f} GB | "
        f"total {TOTAL:.2f} GB | headroom {headroom:.2f} GB -> {'FITS' if headroom > 1.0 else 'TIGHT/RISK'}"
    )
    return headroom


def main() -> None:
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | total {TOTAL:.2f} GB | "
        f"bf16 supported: {torch.cuda.is_bf16_supported()}"
    )
    try:
        if trial("cuda") > 1.0:
            print("\nRECOMMENDATION: all on GPU (bf16/fp16); RAG_RERANKER_DEVICE=cuda.")
            return
        raise torch.cuda.OutOfMemoryError("tight headroom")
    except torch.cuda.OutOfMemoryError:
        print("\nAll-GPU too tight. Retrying with reranker on CPU RAM...")
        trial("cpu")
        print(
            "\nRECOMMENDATION: embedder+LLM on GPU, reranker on CPU RAM; RAG_RERANKER_DEVICE=cpu."
        )


if __name__ == "__main__":
    main()
