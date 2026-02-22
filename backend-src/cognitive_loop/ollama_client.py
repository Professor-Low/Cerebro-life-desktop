"""
Ollama Client for Local LLM Reasoning

Connects to DGX Spark for local LLM inference (configured via OLLAMA_URL env var).
Supports multiple models with automatic thinking extraction.

Supported Models:
- deepseek-r1:32b-qwen-distill-q4_K_M (default) - Best for reasoning
- qwen3:32b - Good general purpose
- deepseek-r1:70b-llama-distill-q4_K_M - Larger, slower, better reasoning
"""

import os
import json
import re
import aiohttp
from typing import AsyncGenerator, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """A chat message with role and content."""
    role: str  # 'system', 'user', 'assistant'
    content: str


@dataclass
class OllamaResponse:
    """Response from Ollama with optional thinking extraction."""
    content: str
    thinking: Optional[str] = None
    model: str = ""
    total_duration_ms: float = 0
    tokens_generated: int = 0
    tokens_per_second: float = 0


# Model configurations
MODEL_CONFIGS = {
    "deepseek-r1:32b-qwen-distill-q4_K_M": {
        "thinking_mode": "natural",  # DeepSeek-R1 reasons naturally
        "think_pattern": r'<think>(.*?)</think>',
        "description": "DeepSeek-R1 32B - Optimized for reasoning tasks"
    },
    "deepseek-r1:70b-llama-distill-q4_K_M": {
        "thinking_mode": "natural",
        "think_pattern": r'<think>(.*?)</think>',
        "description": "DeepSeek-R1 70B - More capable but slower"
    },
    "qwen3:32b": {
        "thinking_mode": "prompted",  # Needs /think suffix
        "think_pattern": r'<think>(.*?)</think>',
        "description": "Qwen3 32B - Good general purpose"
    },
    "qwen3:8b": {
        "thinking_mode": "prompted",
        "think_pattern": r'<think>(.*?)</think>',
        "description": "Qwen3 8B - Fast but less capable"
    },
    "llama3.1:70b": {
        "thinking_mode": "none",  # No special thinking mode
        "think_pattern": None,
        "description": "Llama 3.1 70B - Strong general model"
    }
}


class OllamaClient:
    """
    Async client for Ollama API with streaming support.

    Supports multiple models with automatic thinking extraction:
    - DeepSeek-R1: Natural reasoning with <think> tags
    - Qwen3: Prompted reasoning with /think suffix
    - Others: Standard completion
    """

    OLLAMA_URL = os.environ.get("OLLAMA_URL", "")

    # Default to DeepSeek-R1 for better reasoning
    DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:32b-qwen-distill-q4_K_M")

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or self.OLLAMA_URL
        self.model = model or self.DEFAULT_MODEL
        self._session: Optional[aiohttp.ClientSession] = None

        # Get model config
        self._config = MODEL_CONFIGS.get(self.model, {
            "thinking_mode": "none",
            "think_pattern": r'<think>(.*?)</think>',
            "description": "Unknown model"
        })

        # Compile think pattern if present
        if self._config.get("think_pattern"):
            self._think_re = re.compile(self._config["think_pattern"], re.DOTALL)
        else:
            self._think_re = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout for long generations
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def is_available(self) -> bool:
        """Check if Ollama server is available."""
        if not self.base_url:
            return False
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """List available models."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def set_model(self, model: str):
        """Change the model being used."""
        self.model = model
        self._config = MODEL_CONFIGS.get(model, {
            "thinking_mode": "none",
            "think_pattern": r'<think>(.*?)</think>',
            "description": "Unknown model"
        })
        if self._config.get("think_pattern"):
            self._think_re = re.compile(self._config["think_pattern"], re.DOTALL)
        else:
            self._think_re = None

    async def chat(
        self,
        messages: List[ChatMessage],
        thinking: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> OllamaResponse:
        """
        Send chat request and get complete response.

        Args:
            messages: List of chat messages
            thinking: Enable thinking mode (model-dependent behavior)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            OllamaResponse with content and optional thinking
        """
        # Standalone: no Ollama server — return empty response
        if not self.base_url:
            return OllamaResponse(content="", thinking=None, model="none",
                                  total_duration_ms=0, tokens_generated=0, tokens_per_second=0)

        # Collect full response from stream
        full_content = ""
        response_meta = {}

        async for chunk in self.chat_stream(messages, thinking, temperature, max_tokens):
            if isinstance(chunk, dict):
                if chunk.get("done"):
                    response_meta = chunk
                elif "message" in chunk:
                    full_content += chunk["message"].get("content", "")
            else:
                full_content += str(chunk)

        # DeepSeek-R1 may return thinking in response metadata
        thinking_content = response_meta.get("thinking")

        # Also check for embedded <think> tags in content
        if not thinking_content:
            full_content, thinking_content = self.extract_thinking(full_content)

        return OllamaResponse(
            content=full_content,
            thinking=thinking_content,
            model=response_meta.get("model", self.model),
            total_duration_ms=response_meta.get("total_duration", 0) / 1_000_000,
            tokens_generated=response_meta.get("eval_count", 0),
            tokens_per_second=self._calc_tps(response_meta)
        )

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        thinking: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response from Ollama.

        Yields chunks as they arrive for real-time display.

        Args:
            messages: List of chat messages
            thinking: Enable thinking mode
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Dict chunks from Ollama stream
        """
        if not self.base_url:
            yield {"done": True, "model": "none"}
            return

        session = await self._get_session()

        # Format messages for Ollama
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Apply model-specific thinking mode
        thinking_mode = self._config.get("thinking_mode", "none")

        if thinking_mode == "prompted" and thinking and formatted_messages:
            # Qwen3 uses /think or /no_think at end of prompt
            last_msg = formatted_messages[-1]
            if last_msg["role"] == "user" and "/think" not in last_msg["content"].lower():
                formatted_messages[-1]["content"] = last_msg["content"] + " /think"
        elif thinking_mode == "prompted" and not thinking and formatted_messages:
            last_msg = formatted_messages[-1]
            if last_msg["role"] == "user" and "/no_think" not in last_msg["content"].lower():
                formatted_messages[-1]["content"] = last_msg["content"] + " /no_think"
        # DeepSeek-R1 (thinking_mode="natural") reasons by default, no modification needed

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    yield {"error": f"Ollama error {resp.status}: {error_text}"}
                    return

                async for line in resp.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            yield chunk
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            yield {"error": f"Connection error: {str(e)}"}
        except Exception as e:
            yield {"error": f"Unexpected error: {str(e)}"}

    async def generate(
        self,
        prompt: str,
        thinking: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> OllamaResponse:
        """
        Simple generate endpoint (non-chat).

        Args:
            prompt: The prompt text
            thinking: Enable thinking mode
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            OllamaResponse with content
        """
        # Apply model-specific thinking mode
        thinking_mode = self._config.get("thinking_mode", "none")

        if thinking_mode == "prompted":
            if thinking:
                prompt = prompt + " /think"
            else:
                prompt = prompt + " /no_think"

        session = await self._get_session()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return OllamaResponse(
                        content=f"Error: {error_text}",
                        model=self.model
                    )

                data = await resp.json()
                full_response = data.get("response", "")

                # DeepSeek-R1 returns thinking in a separate field!
                thinking_content = data.get("thinking")

                # Also check for embedded <think> tags as fallback
                if not thinking_content:
                    full_response, thinking_content = self.extract_thinking(full_response)

                return OllamaResponse(
                    content=full_response,
                    thinking=thinking_content,
                    model=data.get("model", self.model),
                    total_duration_ms=data.get("total_duration", 0) / 1_000_000,
                    tokens_generated=data.get("eval_count", 0),
                    tokens_per_second=self._calc_tps(data)
                )

        except Exception as e:
            return OllamaResponse(
                content=f"Error: {str(e)}",
                model=self.model
            )

    def extract_thinking(self, response: str) -> Tuple[str, Optional[str]]:
        """
        Extract thinking blocks from response.

        Supports multiple formats:
        - <think>...</think> (DeepSeek-R1, Qwen3)

        Args:
            response: Full response text

        Returns:
            Tuple of (content without thinking, thinking content or None)
        """
        if self._think_re:
            match = self._think_re.search(response)
            if match:
                thinking = match.group(1).strip()
                content = self._think_re.sub('', response).strip()
                return content, thinking
        return response, None

    def _calc_tps(self, data: dict) -> float:
        """Calculate tokens per second from Ollama response."""
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 1)  # nanoseconds
        if eval_duration > 0:
            return eval_count / (eval_duration / 1_000_000_000)
        return 0.0

    def get_model_info(self) -> dict:
        """Get info about the current model."""
        return {
            "model": self.model,
            "description": self._config.get("description", "Unknown"),
            "thinking_mode": self._config.get("thinking_mode", "none"),
            "base_url": self.base_url
        }


# Convenience function for quick usage
async def quick_think(prompt: str, thinking: bool = True, model: Optional[str] = None) -> str:
    """
    Quick helper to get a response from the LLM.

    Example:
        response = await quick_think("What is the meaning of life?")
        response = await quick_think("Analyze this", model="qwen3:32b")
    """
    client = OllamaClient(model=model)
    try:
        messages = [ChatMessage(role="user", content=prompt)]
        response = await client.chat(messages, thinking=thinking)
        return response.content
    finally:
        await client.close()
