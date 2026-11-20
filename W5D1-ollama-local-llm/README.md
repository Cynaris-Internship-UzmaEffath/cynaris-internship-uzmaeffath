# W5D1: Running LLMs Locally with Ollama

## Objective

Install Ollama, run a local language model, call the Ollama API
using Python, and compare responses from two local LLMs.

## Models Used

- `llama3.2:3b`
- `qwen2.5:3b`

## Technologies

- Python 3.11
- Ollama
- Python `urllib` and `json`
- Ollama local Chat API

## Project Structure

```text
W5D1-ollama-local-llm/
├── ollama_inference.py
├── compare_models.py
├── outputs/
│   ├── first_run_results.md
│   └── model_comparison.md
└── README.md
```

## Tasks Completed

### 1. Ollama Setup

Installed Ollama and downloaded both models. Successfully ran
`llama3.2:3b` locally to test the first inference.

### 2. Python Inference Script

Created `ollama_inference.py` to call the Ollama API using a
custom system prompt.

The script tests five prompts covering:
- Local LLM inference
- Model parameters and prompt parameters
- Local LLMs versus cloud APIs
- Quantisation
- Overfitting

The generated responses are saved in
`outputs/first_run_results.md`.

### 3. Model Comparison

Created `compare_models.py` to ask both models the same three
questions about model parameters, local inference, and
quantisation.

The responses are saved in `outputs/model_comparison.md`.

### 4. Initial Observations

For the quantisation question, `llama3.2:3b` provided a more
detailed explanation, while `qwen2.5:3b` gave a shorter answer.

Both models explained that quantisation can reduce model size
and improve inference efficiency, but may also reduce accuracy.

These are qualitative observations from the responses, not a
formal benchmark of speed or overall model quality.

## How to Run

Make sure Ollama is installed and running.

Check the downloaded models:

```powershell
ollama list
```

Run the first inference script:

```powershell
python W5D1-ollama-local-llm\ollama_inference.py
```

Run the comparison script:

```powershell
python W5D1-ollama-local-llm\compare_models.py
```

Both scripts use the local Ollama API at
`http://localhost:11434/api/chat`.

No cloud API key is required.

## Viva Questions

### 1. What is the difference between a model parameter and a prompt parameter?

A model parameter is a learned value, such as a weight, that
forms part of the model. A prompt parameter is an input or
instruction provided to guide the model's response at inference
time.

### 2. Why use a local LLM instead of a cloud API?

A local LLM provides greater control over execution and can
keep prompts on the local machine. It can also work offline
after setup. However, it requires sufficient hardware resources
and may be less capable than larger cloud-hosted models.

### 3. What is quantisation and how does it affect model quality?

Quantisation represents model values using lower numerical
precision. It can reduce memory usage and improve inference
efficiency, but aggressive quantisation may reduce response
quality. The impact depends on the model and quantisation
method.

## Self-Review Checklist

- [x] Ollama installed and both models downloaded
- [x] First local inference completed
- [x] Python script created with a custom system prompt
- [x] Five prompts tested
- [x] Both models compared using the same three questions
- [x] Output files generated
- [ ] CIA Full Stack Mentor Mode code review completed
- [ ] Minimum two CIA interactions logged
- [ ] Two descriptive commits created
- [ ] Branch pushed and pull request opened

## Limitations

The comparison uses only three questions and qualitative
observations. It does not provide a comprehensive evaluation
of factual accuracy, latency, memory consumption, or general
model performance.
