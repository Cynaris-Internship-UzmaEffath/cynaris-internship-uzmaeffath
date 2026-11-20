# Production RAG API - Monitoring Strategy

## 1. Overview

The Production RAG API uses FastAPI, Docker and a locally hosted Ollama model (llama3.2:3b). Monitoring helps maintain API availability, performance and reliability.

Note: The current implementation provides an Ollama-powered generation endpoint. It does not yet include a document retrieval component.

## 2. Metrics to Monitor

| Metric | What to Monitor | Purpose |
|---|---|---|
| API availability | GET /health response | Detect service outages |
| Request volume | Requests per minute | Identify traffic changes |
| Response latency | Average and 95th percentile latency | Detect slow responses |
| Error rate | HTTP 4xx and 5xx responses | Identify application failures |
| Ollama availability | Connection status and model availability | Detect inference service failures |
| Generation latency | Time taken by /generate | Monitor model performance |
| Container health | Docker status and restart count | Detect container failures |
| Resource usage | CPU and memory consumption | Prevent resource exhaustion |

## 3. Alerting Strategy

| Condition | Alert Threshold | Action |
|---|---|---|
| API unavailable | Health check fails for 1 minute | Check container and application logs |
| Ollama unavailable | Connection failure | Verify Ollama service and model |
| High error rate | More than 5% errors over 5 minutes | Inspect API logs and dependencies |
| High latency | P95 response time above 30 seconds for 5 minutes | Check model load and system resources |
| Container restart | Unexpected restart | Inspect Docker logs and restart reason |
| High memory usage | Above 85% for 5 minutes | Investigate memory usage and workload |

Thresholds are initial operational targets and should be adjusted using actual production traffic.

## 4. Monitoring Tools

- FastAPI health endpoint: /health
- Docker: docker ps, docker stats and docker logs
- Python logging: Uvicorn request and application logs
- GitHub Actions: CI pipeline status and build failures
- Future extension: Prometheus and Grafana for metrics dashboards and alerting

## 5. Logging and Security

- Record request timestamps, endpoint, status code and response duration.
- Record application errors and Ollama connection failures.
- Do not log raw prompts, credentials, API tokens or personal information.
- Restrict access to logs and monitoring dashboards.
- Rotate logs periodically to control storage usage.

## 6. Model Evaluation and Retraining Triggers

The current service uses a pretrained local Ollama model. It does not train or fine-tune the model.

Review model quality when:
- Generated responses become inaccurate or irrelevant.
- Response quality decreases on a fixed evaluation dataset.
- A new Ollama model version becomes available.
- Inference latency or resource consumption increases significantly.

Actions:
- Run a fixed set of evaluation prompts.
- Compare response quality and latency across model versions.
- Update the model only after evaluation.
- If a retrieval component is introduced, monitor document freshness and retrieval quality, and re-index documents when the knowledge base changes.
- Fine-tuning or retraining should be considered only when evaluation shows that prompt or configuration improvements are insufficient.

## 7. Incident Response

1. Check API health using GET /health.
2. Inspect container status using docker ps.
3. Review application logs using docker logs production-rag-api.
4. Verify Ollama availability and model installation.
5. Restart the affected service if required.
6. Retest the root, health and generation endpoints.
7. Document the incident and corrective action.

## 8. Maintenance Schedule

- Daily: Check API health and CI workflow status.
- Weekly: Review error rates, latency and resource usage.
- Monthly: Evaluate model quality and review monitoring thresholds.
- Before model updates: Run regression tests and compare evaluation results.

## 9. Conclusion

This monitoring strategy provides a practical foundation for operating the Dockerised API. Health checks, structured logs, performance metrics, alert thresholds and regular model evaluation help maintain service reliability and support future production improvements.
