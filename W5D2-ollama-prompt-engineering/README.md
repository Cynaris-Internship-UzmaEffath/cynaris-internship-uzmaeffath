# W5D2: Prompt Engineering and System Prompts with Ollama

## Objective
Run local language models, apply custom system prompts, test prompt responses, and compare Llama 3.2 3B with Qwen 2.5 3B.

## Tools Used
- Ollama 0.40.11
- Python
- Llama 3.2 3B
- Qwen 2.5 3B
- Python standard library: urllib, json, pathlib, time

## Implementation
1. Downloaded and tested Llama 3.2 3B locally.
2. Downloaded Qwen 2.5 3B for comparison.
3. Created `ollama_prompt_test.py` to test five prompts with a custom system prompt.
4. Created `model_comparison.py` to test both models using the same three questions.
5. Saved the generated responses and timing data as JSON files in `outputs/`.

## Model Comparison
Both models generated responses for all three questions, giving six successful responses.

| Evaluation aspect | What to observe |
|---|---|
| Clarity | Is the answer easy to understand? |
| Accuracy | Are the technical claims correct? |
| Completeness | Does the answer cover the important points? |
| Examples | Are examples practical and relevant? |
| Response time | Compare the recorded times in seconds. |

### Initial observation
For the REST API security question, both models suggested HTTPS, authentication, and rate limiting. Their explanations and examples differed. The saved JSON results should be reviewed before deciding which model performed better overall.

Response times depend on hardware, model loading, and execution conditions. This small test does not establish that one model is universally better.

## Output Files
- `outputs/prompt_test_results.json`
- `outputs/model_comparison.json`

## Viva Questions

### 1. What is the difference between a model parameter and a prompt parameter?
A model parameter is a learned value, such as a weight, adjusted during training. A prompt parameter is an instruction or input setting that influences how the model responds at inference time. Temperature is an inference option, not a learned model weight.

### 2. Why use a local LLM instead of a cloud API?
A local LLM can work without sending prompts to an external inference provider, offers more control over data handling, and avoids per-request cloud API charges. However, it requires local storage and computing resources, and may be slower or less capable than larger hosted models.

### 3. What is quantisation, and how does it affect model quality?
Quantisation represents model weights using fewer bits. It reduces memory requirements and can improve inference efficiency, but may reduce accuracy or response quality depending on the quantisation method and task.

## Self-Review Checklist
- [ ] Ollama installed and both models downloaded
- [ ] Five system-prompt tests completed
- [ ] Three identical questions tested on both models
- [ ] JSON outputs inspected
- [ ] Comparison observations documented
- [ ] Code reviewed using CIA Full Stack Mentor Mode
- [ ] At least two CIA interactions completed and logged
- [ ] Evidence screenshot captured
- [ ] At least two descriptive commits created
- [ ] Branch pushed and pull request raised
