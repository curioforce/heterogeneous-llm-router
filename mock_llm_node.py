#!/usr/bin/env python3
"""
mock_llm_node.py

A fake OpenAI-compatible LLM backend.

Use this to test the router without actual Apple Silicon or NVIDIA GPU nodes.

Examples:
    python mock_llm_node.py --name apple-mock-1 --hardware apple_silicon --port 8001
    python mock_llm_node.py --name nvidia-mock-1 --hardware nvidia_gpu --port 8002
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512


def create_app(node_name: str, hardware_type: str) -> FastAPI:
    """Create one mock LLM node app."""

    app = FastAPI(title=f"Mock LLM Node - {node_name}")

    @app.get("/health")
    def health() -> Dict[str, str]:
        """
        The router expects this endpoint.

        Real nodes should return unhealthy if model serving, GPU, Metal backend,
        or queue processing has a problem.
        """
        return {
            "status": "ok",
            "node_name": node_name,
            "hardware_type": hardware_type,
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        Fake OpenAI-compatible chat completion endpoint.

        This does not run a real model. It returns a pretend response so you can
        test routing, model selection, health checks, and fallback behavior.
        """

        user_messages = [
            message.content
            for message in request.messages
            if message.role == "user"
        ]
        last_user_message = user_messages[-1] if user_messages else ""

        # Simulate a tiny amount of inference work.
        time.sleep(0.3)

        return {
            "id": f"mock-{node_name}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            f"Mock response from {node_name} "
                            f"running on {hardware_type}. "
                            f"Selected model: {request.model}. "
                            f"You asked: {last_user_message}"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Human-readable node name")
    parser.add_argument("--hardware", required=True, choices=["apple_silicon", "nvidia_gpu"])
    parser.add_argument("--port", type=int, required=True)

    args = parser.parse_args()

    app = create_app(node_name=args.name, hardware_type=args.hardware)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
