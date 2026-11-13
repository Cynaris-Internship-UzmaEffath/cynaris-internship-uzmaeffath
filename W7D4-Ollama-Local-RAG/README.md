# W7D4 – LlamaIndex + Ollama: Local RAG

## 1. Objective

The objective of this task was to set up and test local Large Language Models (LLMs) using Ollama, communicate with the Ollama API through Python, and compare the responses of two locally running models.

## 2. Tools and Technologies

- Python 3.11.9
- Ollama 0.40.16
- llama3.2:3b
- qwen2.5:3b
- JSON
- Python urllib
- Visual Studio Code
- Git and GitHub

## 3. Environment Setup

Ollama was already installed in the development environment.

The following models were available:

- llama3.2:3b
- qwen2.5:3b
- all-minilm:latest

Both LLMs were tested locally using the Ollama command line.

## 4. Python Ollama API Integration

A Python script named `ollama_api_test.py` was created to communicate with the local Ollama REST API.

The script includes:

- A custom system prompt.
- Five test questions.
- HTTP requests using Python urllib.
- JSON request and response handling.
- Response-time measurement.

All five prompts executed successfully using llama3.2:3b.

## 5. Model Comparison

A second Python script named `model_comparison.py` was developed to compare llama3.2:3b and qwen2.5:3b.

Both models received the same three questions:

1. What is Retrieval-Augmented Generation (RAG)?
2. How does image classification work in Machine Learning?
3. How can AI help detect crop diseases?

The script measured response times and saved the generated answers in `comparison_results.json`.

### Response-Time Comparison

| Question               | llama3.2:3b | qwen2.5:3b |
| ---------------------- | ----------: | ---------: |
| RAG                    |   31.43 sec |  42.98 sec |
| Image Classification   |   45.31 sec |  43.15 sec |
| Crop Disease Detection |   31.74 sec |  73.95 sec |
| Average                |   36.16 sec |  53.36 sec |

### Observations

- llama3.2:3b produced detailed explanations and had a lower average response time.
- qwen2.5:3b produced conversational explanations with practical examples.
- Both models generated relevant responses to the three questions.
- llama3.2:3b was approximately 32.2% faster in this experiment.

These results are specific to the current machine and test execution.

## 6. Relevance to AgroMind Crop Disease Detection

The model comparison included a question about AI-based crop disease detection.

The responses explained how AI can use image analysis, computer vision, and machine learning to identify disease symptoms in crops.

This knowledge is useful for understanding how a local LLM can support an agricultural knowledge assistant.

In the AgroMind project, image classification can identify crop diseases, while a RAG-based assistant can retrieve relevant information from agricultural documents and explain disease symptoms and possible interventions.

## 7. Project Structure

```text
W7D4-Ollama-Local-RAG/
│
├── ollama_api_test.py
├── model_comparison.py
├── comparison_results.json
└── README.md
```

## 8. Learning Outcomes

- Learned how to run local LLMs using Ollama.
- Understood local inference.
- Learned how to communicate with Ollama through its REST API.
- Implemented custom system prompts.
- Tested multiple prompts through Python.
- Compared two locally hosted language models.
- Measured and documented response times.
- Understood how local LLMs can support RAG applications.

## 9. Conclusion

Successfully configured and tested local LLM inference using Ollama. Developed Python scripts for API-based interaction and model comparison. Both models completed the comparison tasks, and the results were saved in JSON format for further analysis.

**Note:** This practical covers local LLM inference and model comparison. Complete document indexing and retrieval integration are separate RAG implementation steps.
