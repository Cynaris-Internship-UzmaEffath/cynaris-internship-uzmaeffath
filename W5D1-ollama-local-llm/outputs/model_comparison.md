# W5D1: Local LLM Comparison

Both models received the same system prompt and questions.

## Question 1

What is the difference between a model parameter and a prompt parameter?

### llama3.2:3b

In the context of machine learning models, especially those using natural language processing, there are two types of parameters to consider: model parameters and prompt parameters.

**Model parameters** are the weights and biases of a model that are learned during training. These parameters are specific to the model itself and are used to predict the output of the model.

**Prompt parameters**, on the other hand, are input parameters that are used to fine-tune the model's output. These parameters are specific to the task or question being asked, and are used to guide the model's response.

### qwen2.5:3b

A model parameter is fixed within a model's architecture and determines how data is processed. A prompt parameter, on the other hand, is defined externally and guides the model's output. It's like setting up a scenario or question a model answers based on its internal parameters.

## Question 2

Why use a local LLM instead of a cloud API?

### llama3.2:3b

Using a local Large Language Model (LLM) instead of a cloud API has several advantages:

1. **Data Security**: Your data is stored locally, not in the cloud, ensuring greater control over data protection.
2. **Faster Processing**: Local models can process data faster since they don't require internet connectivity or wait for API responses.
3. **Reduced Latency**: You can make predictions instantly without relying on cloud API latency.
4. **Increased Efficiency**: Training local models can be more efficient, especially for large-scale applications.
5. **Better Model Performance**: Local models can be optimized for your specific use case, leading to better performance.

However, local models require significant resources and expertise to train and maintain.

### qwen2.5:3b

Using a local Language Model (LLM) can offer faster response times and better privacy. It's especially useful for real-time applications where delay is critical. Locals also don't rely on internet connections, avoiding potential outages. However, they require more initial setup and maintenance.

## Question 3

What is quantisation, and how can it affect model quality?

### llama3.2:3b

Quantization is a process that reduces the precision of model weights and activations from floating-point numbers (like 32-bit or 64-bit) to smaller integers (usually 8-bit or 16-bit). This affects model quality in several ways:

* Reduced precision can lead to smaller model sizes and faster inference times, but may also decrease accuracy and performance.
* Quantization can cause loss of information and distortion, particularly for models that rely on high-precision calculations.
* In some cases, quantization can be used to improve model performance, but it requires careful tuning and optimization.

### qwen2.5:3b

Quantization is a technique used to reduce the number of bits used to represent numbers in a model. This makes the model smaller and faster, but can reduce accuracy. It affects model quality by potentially leading to less precise predictions.
