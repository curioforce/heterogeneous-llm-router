#!/usr/bin/env python3
"""
llm_router.py

Reference implementation of a simple LLM request router.

This router accepts chat-completion requests, automatically chooses a model,
and then routes the request to either Apple Silicon based nodes or NVIDIA GPU
based nodes.

The code is intentionally clear and heavily commented so it can be used as a
reference starting point.

Example flow:
    Client sends messages only, without specifying a model.
        ↓
    Router estimates task size / complexity.
        ↓
    Router selects a model.
        ↓
    Router selects Apple Silicon or NVIDIA GPU backend.
        ↓
    Router forwards the request to that backend.

Run:
    uvicorn llm_router:app --host 0.0.0.0 --port 9000 --reload
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("llm-router")


# -----------------------------------------------------------------------------
# Hardware and request types
# -----------------------------------------------------------------------------

class HardwareType(str, Enum):
    """Hardware category for an inference node."""

    APPLE_SILICON = "apple_silicon"
    NVIDIA_GPU = "nvidia_gpu"


class RequestPriority(str, Enum):
    """
    Priority level for incoming requests.

    You can use this to send important traffic to stronger/faster nodes.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


# -----------------------------------------------------------------------------
# Backend node definition
# -----------------------------------------------------------------------------

@dataclass
class BackendNode:
    """
    Represents one inference server.

    In real deployments, these servers might run:
    - MLX / llama.cpp Metal / Ollama on Apple Silicon
    - vLLM / TensorRT-LLM / TGI / SGLang on NVIDIA GPUs

    The router only assumes that every backend exposes:
    - GET  /health
    - POST /v1/chat/completions
    """

    name: str
    hardware_type: HardwareType
    base_url: str

    # Models this node can serve.
    supported_models: List[str]

    # Maximum total context supported by this node.
    max_context_tokens: int

    # Rough capacity score. Higher means more capable.
    # This is a simple placeholder for real metrics such as tokens/sec,
    # available memory, queue depth, and observed latency.
    capacity_score: int

    # Dynamic state tracked by the router.
    active_requests: int = 0
    healthy: bool = True
    last_health_check: float = field(default_factory=time.time)

    def chat_completions_url(self) -> str:
        """Return the OpenAI-compatible chat completion endpoint."""
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    def health_url(self) -> str:
        """Return the backend health endpoint."""
        return f"{self.base_url.rstrip('/')}/health"

    def load_score(self) -> float:
        """
        Compute a simple load score.

        Lower is better.

        Production routers should use richer live metrics, such as:
        - GPU/Metal memory usage
        - Request queue depth
        - Recent tokens per second
        - KV cache pressure
        - Recent latency
        - Failure rate
        """
        return self.active_requests / max(self.capacity_score, 1)


# -----------------------------------------------------------------------------
# API request and response models
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """One OpenAI-compatible chat message."""

    role: str
    content: str


class LLMRequest(BaseModel):
    """
    Incoming request to the router.

    The client may omit `model`.
    If omitted, the router automatically chooses the model.
    """

    model: Optional[str] = None
    messages: List[ChatMessage]

    temperature: float = 0.7
    max_tokens: int = 512

    priority: RequestPriority = RequestPriority.NORMAL

    # Optional hard routing preference.
    # If provided, candidates are filtered to this hardware type.
    preferred_hardware: Optional[HardwareType] = None

    # When true, router may retry another backend if the first one fails.
    allow_fallback: bool = True


class LLMResponse(BaseModel):
    """Response returned by the router."""

    selected_model: str
    backend_name: str
    backend_hardware: HardwareType
    response: Dict[str, Any]


# -----------------------------------------------------------------------------
# Backend inventory
# -----------------------------------------------------------------------------
#
# This local inventory points to mock nodes so you can test without real hardware.
#
# Start these mock nodes first:
#
#   python mock_llm_node.py --name apple-mock-1 --hardware apple_silicon --port 8001
#   python mock_llm_node.py --name nvidia-mock-1 --hardware nvidia_gpu --port 8002
#
# For real deployment, replace these base_url values with actual hostnames/IPs.

BACKENDS: List[BackendNode] = [
    BackendNode(
        name="apple-mock-1",
        hardware_type=HardwareType.APPLE_SILICON,
        base_url="http://localhost:8001",
        supported_models=[
            "llama-3.1-8b",
            "mistral-7b",
            "qwen2.5-7b",
        ],
        max_context_tokens=32_000,
        capacity_score=4,
    ),
    BackendNode(
        name="nvidia-mock-1",
        hardware_type=HardwareType.NVIDIA_GPU,
        base_url="http://localhost:8002",
        supported_models=[
            "llama-3.1-8b",
            "llama-3.1-70b",
            "qwen2.5-32b",
            "qwen2.5-72b",
        ],
        max_context_tokens=128_000,
        capacity_score=32,
    ),
]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def estimate_tokens(messages: List[ChatMessage]) -> int:
    """
    Very rough token estimator.

    Approximation:
        1 token ~= 4 characters of English text

    This is not accurate for Japanese, code-heavy text, or every tokenizer.
    In production, use the tokenizer for the selected model.
    """

    total_chars = sum(len(message.content) for message in messages)
    return max(1, total_chars // 4)


def choose_model_for_request(request: LLMRequest) -> str:
    """
    Automatically choose a model based on the request.

    This is a reference policy, not a universal truth.

    Example policy:
    - Explicit model from client: respect it.
    - High priority: use strongest available model.
    - Long or complex request: use stronger NVIDIA-oriented model.
    - Simple/default request: use cheaper/faster small model.
    """

    # Respect explicitly requested model if provided.
    if request.model:
        return request.model

    prompt_tokens = estimate_tokens(request.messages)
    text = "\n".join(message.content.lower() for message in request.messages)

    # Simple keyword-based task classifier.
    # Replace this with a classifier model, policy engine, or rules table later.
    complex_task_keywords = [
        "analyze",
        "analysis",
        "strategy",
        "deep dive",
        "architecture",
        "business plan",
        "legal",
        "financial",
        "compare",
        "benchmark",
        "research",
        "debug",
        "optimize",
        "refactor",
        "write code",
        "production",
        "investor memo",
    ]

    is_complex_task = any(keyword in text for keyword in complex_task_keywords)
    is_long_prompt = prompt_tokens > 4_000

    if request.priority == RequestPriority.HIGH:
        return "qwen2.5-72b"

    if is_long_prompt or is_complex_task:
        return "qwen2.5-32b"

    # Default cheap/fast model.
    # With the included routing policy, this should usually land on Apple Silicon.
    return "llama-3.1-8b"


def health_check_backend(node: BackendNode, timeout_seconds: float = 2.0) -> bool:
    """
    Check whether a backend node is healthy.

    Each worker should expose:
        GET /health -> {"status": "ok"}
    """

    try:
        response = requests.get(node.health_url(), timeout=timeout_seconds)
        response.raise_for_status()

        data = response.json()
        node.healthy = data.get("status") == "ok"

    except Exception as exc:
        logger.warning("Health check failed for %s: %s", node.name, exc)
        node.healthy = False

    node.last_health_check = time.time()
    return node.healthy


def refresh_health_checks(max_age_seconds: float = 15.0) -> None:
    """
    Refresh health checks whose status is stale.

    This avoids health-checking every backend for every request.
    """

    now = time.time()

    for node in BACKENDS:
        health_age = now - node.last_health_check
        if health_age > max_age_seconds:
            health_check_backend(node)


def find_candidate_backends(request: LLMRequest) -> List[BackendNode]:
    """
    Return all healthy backends that can serve the selected model and context.
    """

    if not request.model:
        raise ValueError("request.model must be selected before finding backends")

    estimated_prompt_tokens = estimate_tokens(request.messages)
    estimated_total_tokens = estimated_prompt_tokens + request.max_tokens

    candidates: List[BackendNode] = []

    for node in BACKENDS:
        if not node.healthy:
            continue

        if request.model not in node.supported_models:
            continue

        if estimated_total_tokens > node.max_context_tokens:
            continue

        if request.preferred_hardware is not None:
            if node.hardware_type != request.preferred_hardware:
                continue

        candidates.append(node)

    return candidates


def route_score(node: BackendNode, request: LLMRequest) -> float:
    """
    Compute routing score.

    Lower score is better.

    Policy:
    - Small normal/low-priority requests prefer Apple Silicon.
    - Large models prefer NVIDIA GPU.
    - High-priority requests prefer NVIDIA GPU.
    - Current load still matters.
    """

    score = 0.0

    # Current load is always important.
    score += node.load_score() * 100.0

    if not request.model:
        raise ValueError("request.model must be selected before scoring")

    model_name = request.model.lower()

    large_model_keywords = [
        "30b",
        "32b",
        "34b",
        "65b",
        "70b",
        "72b",
        "mixtral",
        "deepseek",
    ]

    small_model_keywords = [
        "3b",
        "7b",
        "8b",
        "9b",
        "small",
        "mini",
    ]

    is_large_model = any(keyword in model_name for keyword in large_model_keywords)
    is_small_model = any(keyword in model_name for keyword in small_model_keywords)

    # Large models strongly prefer NVIDIA.
    if is_large_model:
        if node.hardware_type == HardwareType.NVIDIA_GPU:
            score -= 30.0
        else:
            score += 30.0

    # Small normal/low-priority models prefer Apple Silicon.
    if is_small_model and request.priority != RequestPriority.HIGH:
        if node.hardware_type == HardwareType.APPLE_SILICON:
            score -= 25.0
        else:
            score += 10.0

    # High-priority traffic prefers NVIDIA.
    if request.priority == RequestPriority.HIGH:
        if node.hardware_type == HardwareType.NVIDIA_GPU:
            score -= 25.0
        else:
            score += 15.0

    # Low-priority traffic should preserve expensive NVIDIA capacity.
    if request.priority == RequestPriority.LOW:
        if node.hardware_type == HardwareType.APPLE_SILICON:
            score -= 15.0
        else:
            score += 10.0

    # Capacity is only a tie-breaker. Do not let it overpower policy.
    score -= node.capacity_score * 0.05

    # Jitter avoids selecting the same node forever when scores are nearly equal.
    score += random.uniform(0, 0.5)

    return score


def choose_backend(request: LLMRequest) -> BackendNode:
    """
    Select the best backend for the request.

    This function also chooses a model if the client did not provide one.
    """

    # Auto-select model before looking for candidate backends.
    request.model = choose_model_for_request(request)

    refresh_health_checks()

    candidates = find_candidate_backends(request)

    if not candidates:
        raise HTTPException(
            status_code=503,
            detail=f"No healthy backend can serve selected model: {request.model}",
        )

    model_name = request.model.lower()
    small_model_keywords = ["3b", "7b", "8b", "9b", "small", "mini"]
    is_small_model = any(keyword in model_name for keyword in small_model_keywords)

    # Deterministic shortcut:
    # For small normal/low-priority requests, use Apple Silicon first if available.
    if is_small_model and request.priority != RequestPriority.HIGH:
        apple_candidates = [
            node for node in candidates
            if node.hardware_type == HardwareType.APPLE_SILICON
        ]

        if apple_candidates:
            selected = sorted(apple_candidates, key=lambda node: node.load_score())[0]
            logger.info(
                "Auto-selected model=%s and Apple Silicon backend=%s",
                request.model,
                selected.name,
            )
            return selected

    ranked = sorted(candidates, key=lambda node: route_score(node, request))
    selected = ranked[0]

    logger.info(
        "Auto-selected model=%s backend=%s hardware=%s active_requests=%s",
        request.model,
        selected.name,
        selected.hardware_type,
        selected.active_requests,
    )

    return selected


def forward_request_to_backend(node: BackendNode, request: LLMRequest) -> Dict[str, Any]:
    """
    Forward the request to the selected backend.

    The backend is expected to expose an OpenAI-compatible endpoint:
        POST /v1/chat/completions
    """

    if not request.model:
        raise ValueError("Cannot forward request without selected model")

    payload = {
        "model": request.model,
        "messages": [message.model_dump() for message in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }

    node.active_requests += 1

    try:
        response = requests.post(
            node.chat_completions_url(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    finally:
        node.active_requests -= 1


def forward_with_fallback(primary_node: BackendNode, request: LLMRequest) -> Tuple[BackendNode, Dict[str, Any]]:
    """
    Try the selected backend first.

    If it fails and fallback is allowed, try another suitable backend.
    """

    try:
        result = forward_request_to_backend(primary_node, request)
        return primary_node, result

    except Exception as first_error:
        logger.warning(
            "Primary backend failed: backend=%s error=%s",
            primary_node.name,
            first_error,
        )
        primary_node.healthy = False

        if not request.allow_fallback:
            raise HTTPException(
                status_code=502,
                detail=f"Backend {primary_node.name} failed and fallback is disabled.",
            )

    fallback_candidates = [
        node
        for node in find_candidate_backends(request)
        if node.name != primary_node.name
    ]
    fallback_candidates = sorted(fallback_candidates, key=lambda node: route_score(node, request))

    for fallback_node in fallback_candidates:
        try:
            logger.info("Trying fallback backend=%s", fallback_node.name)
            result = forward_request_to_backend(fallback_node, request)
            return fallback_node, result

        except Exception as fallback_error:
            logger.warning(
                "Fallback backend failed: backend=%s error=%s",
                fallback_node.name,
                fallback_error,
            )
            fallback_node.healthy = False

    raise HTTPException(status_code=502, detail="All suitable backends failed.")


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="LLM Hardware Router",
    description="Auto-selects a model and routes requests to Apple Silicon or NVIDIA GPU nodes.",
    version="0.2.0",
)


@app.get("/health")
def router_health() -> Dict[str, Any]:
    """Health endpoint for the router itself."""

    healthy_backends = [node.name for node in BACKENDS if node.healthy]

    return {
        "status": "ok",
        "healthy_backend_count": len(healthy_backends),
        "healthy_backends": healthy_backends,
    }


@app.get("/backends")
def list_backends() -> Dict[str, Any]:
    """Show known backend nodes and their current router-side status."""

    return {
        "backends": [
            {
                "name": node.name,
                "hardware_type": node.hardware_type,
                "base_url": node.base_url,
                "supported_models": node.supported_models,
                "max_context_tokens": node.max_context_tokens,
                "capacity_score": node.capacity_score,
                "active_requests": node.active_requests,
                "healthy": node.healthy,
                "load_score": node.load_score(),
            }
            for node in BACKENDS
        ]
    }


@app.post("/v1/chat/completions", response_model=LLMResponse)
def chat_completions(request: LLMRequest) -> LLMResponse:
    """
    Main chat completion endpoint.

    The client may omit `model`.
    The router will choose the model and backend automatically.
    """

    selected_node = choose_backend(request)

    final_node, backend_response = forward_with_fallback(
        primary_node=selected_node,
        request=request,
    )

    if not request.model:
        raise HTTPException(status_code=500, detail="Model selection failed unexpectedly.")

    return LLMResponse(
        selected_model=request.model,
        backend_name=final_node.name,
        backend_hardware=final_node.hardware_type,
        response=backend_response,
    )


# -----------------------------------------------------------------------------
# Local development entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "llm_router:app",
        host="0.0.0.0",
        port=9000,
        reload=True,
    )
