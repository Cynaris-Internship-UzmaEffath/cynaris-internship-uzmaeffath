\# LLM Routing Strategy — Local vs API



\## 1. Objective



The objective is to reduce LLM inference costs while maintaining response quality, privacy, and performance in the local Q\&A application.



\## 2. Local Model vs API



| Criteria | Local LLM (Ollama) | Cloud API |

|---|---|---|

| Cost | No per-token API charges; uses local hardware | Pay per token |

| Privacy | Data remains on the local machine | Data is sent to an external provider |

| Internet | Not required after model download | Required |

| Latency | Depends on local hardware | Depends on network and provider |

| Scalability | Limited by local resources | Can scale with provider capacity |

| Availability | Depends on local system | Depends on API availability |



\## 3. Proposed Routing Strategy



\### Route 1 — Semantic Cache



\- Check the semantic cache before generating a response.

\- Use cosine similarity with a threshold of 0.92.

\- Return the cached answer when a sufficiently similar question is found.

\- Avoid unnecessary model inference for cache hits.



\### Route 2 — Local LLM



Use Ollama as the default inference option.



\- Use Llama 3.2 (3B) for general question answering.

\- Use Qwen 2.5 (3B) as an alternative model.

\- Retrieve relevant document chunks from ChromaDB.

\- Compress unnecessary prompt content when appropriate.

\- Track token usage for every request.



\### Route 3 — Cloud API Fallback



An external API can be considered when:



\- The local model cannot produce an acceptable answer.

\- A task requires capabilities unavailable in the local model.

\- Local hardware resources are insufficient.

\- The user explicitly permits external processing.



API usage must be optional and require configured credentials. Sensitive document content should not be sent externally without permission.



\## 4. Cost Optimisation Techniques



1\. Semantic caching to avoid repeated inference.

2\. Prompt compression to reduce input tokens.

3\. Retrieval of only relevant document chunks.

4\. Selection of smaller local models for routine questions.

5\. Token usage monitoring and cost estimation.

6\. API fallback only when necessary.



\## 5. Proposed Workflow



User Question → Semantic Cache → ChromaDB Retrieval → Prompt Compression → Local Ollama Model → Response



If local inference is insufficient and external processing is permitted, an optional API fallback can be considered.



\## 6. Conclusion



A local-first routing strategy is suitable for this Q\&A application because it reduces recurring API expenses and keeps document processing on the local machine. Semantic caching and prompt compression can further improve efficiency. Cloud APIs remain an optional fallback for tasks that require additional capabilities.
