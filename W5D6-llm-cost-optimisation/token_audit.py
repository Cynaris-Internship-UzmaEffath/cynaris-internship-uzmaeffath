import tiktoken
import pandas as pd
from pathlib import Path

# Use a consistent tokenizer for estimating token counts.
# This is an estimate, not Ollama's exact tokenizer.
encoding = tiktoken.get_encoding("cl100k_base")

base_prompt = """
You are a helpful local AI assistant.
Answer the user's question using the provided context.
Keep your response accurate, clear, and concise.
If the context does not contain the answer, say so.
Do not invent facts or provide unsupported information.
"""

prompt_sizes = {
    "Small": 1,
    "Medium": 5,
    "Large": 10,
    "Very Large": 20,
    "Extra Large": 40
}

results = []

for size_name, repetitions in prompt_sizes.items():
    prompt = base_prompt * repetitions
    token_count = len(encoding.encode(prompt))

    results.append({
        "Prompt Size": size_name,
        "Characters": len(prompt),
        "Token Count (Estimated)": token_count
    })

    print(f"{size_name}: {token_count} tokens")

df = pd.DataFrame(results)

output_path = Path("outputs") / "token_audit.csv"
df.to_csv(output_path, index=False)

print("\nToken audit completed!")
print(df.to_string(index=False))
print(f"\nResults saved to: {output_path}")
