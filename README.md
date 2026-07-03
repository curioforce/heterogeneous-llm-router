# LLM Hardware Router

A reference Python project that automatically chooses an LLM model and routes the request to either Apple Silicon based nodes or NVIDIA GPU based nodes.

This package includes:

- `llm_router.py` - FastAPI router that auto-selects model and backend hardware
- `mock_llm_node.py` - fake OpenAI-compatible LLM node for local testing
- `requirements.txt` - Python dependencies

The default setup runs with mock local nodes, so you do not need real Apple Silicon or NVIDIA GPU machines to test routing.

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Start the mock Apple Silicon node

Terminal 1:

```bash
python mock_llm_node.py --name apple-mock-1 --hardware apple_silicon --port 8001
```

## 3. Start the mock NVIDIA GPU node

Terminal 2:

```bash
python mock_llm_node.py --name nvidia-mock-1 --hardware nvidia_gpu --port 8002
```

## 4. Start the router

Terminal 3:

```bash
uvicorn llm_router:app --host 0.0.0.0 --port 9000 --reload
```

Or:

```bash
python llm_router.py
```

## 5. Test simple request without specifying model

This should auto-select `llama-3.1-8b` and route to Apple Silicon.

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain Apple Silicon routing."
      }
    ],
    "temperature": 0.7,
    "max_tokens": 300,
    "priority": "normal"
  }'
```

Expected shape:

```json
{
  "selected_model": "llama-3.1-8b",
  "backend_name": "apple-mock-1",
  "backend_hardware": "apple_silicon",
  "response": {
    "choices": [
      {
        "message": {
          "content": "Mock response from apple-mock-1..."
        }
      }
    ]
  }
}
```

## 6. Test complex request without specifying model

This should auto-select `qwen2.5-32b` and route to NVIDIA.

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Analyze this business strategy and create a production-ready architecture."
      }
    ],
    "temperature": 0.7,
    "max_tokens": 1000,
    "priority": "normal"
  }'
```

## 7. Test high-priority request

This should auto-select `qwen2.5-72b` and route to NVIDIA.

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Write a detailed investor memo."
      }
    ],
    "temperature": 0.7,
    "max_tokens": 1500,
    "priority": "high"
  }'
```

## 8. Inspect backend status

```bash
curl http://localhost:9000/backends
```

## Routing policy summary

The included reference policy is intentionally simple:

| Request type | Selected model | Preferred hardware |
|---|---:|---|
| Simple / normal | `llama-3.1-8b` | Apple Silicon |
| Complex / analysis / production / debug | `qwen2.5-32b` | NVIDIA GPU |
| High priority | `qwen2.5-72b` | NVIDIA GPU |

## Moving from mock nodes to real nodes

Replace the `BACKENDS` list in `llm_router.py` with your real node URLs.

Each real node should expose:

```text
GET  /health
POST /v1/chat/completions
```

As long as your worker nodes expose an OpenAI-compatible `/v1/chat/completions` endpoint, they can use different runtimes internally, such as MLX, llama.cpp Metal, Ollama, vLLM, TensorRT-LLM, TGI, or SGLang.

## Production ideas

Useful next additions:

- API key authentication
- Streaming response forwarding
- Redis-backed backend registry
- Prometheus metrics
- Real tokenizer-based token counting
- Per-user or per-tenant rate limiting
- Cost-aware routing
- GPU memory-aware scheduling
- Circuit breakers and exponential backoff
- Model aliases such as `fast`, `smart`, `cheap`, or `private`
