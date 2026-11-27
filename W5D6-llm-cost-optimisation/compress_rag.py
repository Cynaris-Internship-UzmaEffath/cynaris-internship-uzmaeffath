from pathlib import Path
import json
import time
import tiktoken
from llmlingua import PromptCompressor

encoding = tiktoken.get_encoding("cl100k_base")

prompt_path = Path("outputs/original_rag_prompt.txt")
prompt = prompt_path.read_text(encoding="utf-8")

original_tokens = len(encoding.encode(prompt))

print(f"Original tokens: {original_tokens}")
print("Loading LLMLingua-2 compression model...")

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="cpu"
)
start_time = time.time()

result = compressor.compress_prompt(
    prompt,
    rate=0.4,
    force_tokens=["\n", ".", "?", "!"]
)

compressed_prompt = result["compressed_prompt"]
compressed_tokens = len(encoding.encode(compressed_prompt))

compression_time = round(time.time() - start_time, 2)
reduction = (1 - compressed_tokens / original_tokens) * 100

Path("outputs/compressed_rag_prompt.txt").write_text(
    compressed_prompt, encoding="utf-8"
)

report = {
    "original_tokens": original_tokens,
    "compressed_tokens": compressed_tokens,
    "requested_compression_rate": 0.4,
    "actual_token_reduction_percent": round(reduction, 2),
    "compression_time_seconds": compression_time
}

Path("outputs/compression_results.json").write_text(
    json.dumps(report, indent=4),
    encoding="utf-8"
)

print("\nCompression completed!")
print(json.dumps(report, indent=4))
print("\nSaved compressed prompt and results in outputs/")
