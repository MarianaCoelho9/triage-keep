"""MedGemma GGUF Llama.cpp implementation."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from .base import (
    BaseLLMModel,
    GenerationOptions,
    LLMContextBudgetExceededError,
    LLMDecodeError,
    LLMGenerationError,
    apply_stop_sequences,
)


@dataclass(frozen=True)
class MedGemmaLlamaCppConfig:
    """Configuration for llama.cpp backend."""

    gguf_path: Path
    n_ctx: int = 4096
    n_threads: int = 0
    n_batch: int = 128
    max_new_tokens: int = 320
    gpu_layers: int = -1
    context_margin: int = 64

    @classmethod
    def from_env(cls) -> "MedGemmaLlamaCppConfig":
        gguf_path_raw = os.getenv("MEDGEMMA_GGUF_PATH", "")
        if not gguf_path_raw:
            raise ValueError(
                "MEDGEMMA_GGUF_PATH is required for llama.cpp backend."
            )
        gguf_path = Path(gguf_path_raw)
        if not gguf_path.exists():
            raise FileNotFoundError(
                f"MEDGEMMA_GGUF_PATH does not exist: {gguf_path}"
            )

        return cls(
            gguf_path=gguf_path,
            n_ctx=_get_int_env("MEDGEMMA_N_CTX", 4096),
            n_threads=_get_int_env("MEDGEMMA_N_THREADS", 0),
            n_batch=_get_int_env("MEDGEMMA_N_BATCH", 128),
            max_new_tokens=_get_int_env("MEDGEMMA_MAX_NEW_TOKENS", 320),
            gpu_layers=_get_int_env("MEDGEMMA_GPU_LAYERS", -1),
            context_margin=_get_int_env("MEDGEMMA_CONTEXT_MARGIN", 64),
        )


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError as err:
        raise ValueError(f"{name} must be an integer") from err


class MedGemmaLlamaCppService(BaseLLMModel):
    """Local MedGemma using llama.cpp (GGUF)."""

    def __init__(self) -> None:
        print("[MedGemma llama.cpp] Loading model...")
        self.config = MedGemmaLlamaCppConfig.from_env()
        self._generate_lock = threading.Lock()

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as err:
            raise ImportError(
                "llama_cpp is required for medgemma_llamacpp backend. "
                "Install llama-cpp-python."
            ) from err

        n_threads = self.config.n_threads or (os.cpu_count() or 1)

        self.client = Llama(
            model_path=str(self.config.gguf_path),
            n_ctx=self.config.n_ctx,
            n_threads=n_threads,
            n_batch=self.config.n_batch,
            n_gpu_layers=self.config.gpu_layers,
            verbose=False,
        )

        print("[MedGemma llama.cpp] Model loaded.")

    @staticmethod
    def _is_grammar_enabled() -> bool:
        raw = os.getenv("MEDGEMMA_ENABLE_JSON_GRAMMAR", "true").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _is_context_overflow_error(message: str) -> bool:
        lowered = message.lower()
        return (
            "requested tokens" in lowered and "exceed context window" in lowered
        ) or "context window" in lowered

    @staticmethod
    def _is_decode_error(message: str) -> bool:
        lowered = message.lower()
        return "llama_decode returned -1" in lowered or "decode" in lowered

    def _estimate_prompt_tokens(self, prompt: str) -> int:
        try:
            token_ids = self.client.tokenize(prompt.encode("utf-8"))
            return len(token_ids)
        except Exception:
            # Conservative fallback when tokenizer is not exposed.
            return max(1, len(prompt) // 4)

    def _resolve_max_tokens(self, prompt: str, requested_max_tokens: int) -> int:
        prompt_tokens = self._estimate_prompt_tokens(prompt)
        available = self.config.n_ctx - prompt_tokens - self.config.context_margin
        if available < 1:
            raise LLMContextBudgetExceededError(
                "Prompt exceeds llama.cpp context budget."
            )
        if requested_max_tokens < 1:
            raise LLMContextBudgetExceededError(
                "Requested output tokens must be positive."
            )
        return min(requested_max_tokens, available)

    def _build_grammar(self, options: GenerationOptions) -> object | None:
        if options.json_schema is None and options.grammar is None:
            return None
        if not self._is_grammar_enabled():
            return None
        try:
            from llama_cpp import LlamaGrammar  # type: ignore
        except Exception:
            return None

        if options.json_schema is not None:
            schema_json = json.dumps(options.json_schema)
            return LlamaGrammar.from_json_schema(schema_json, verbose=False)
        if options.grammar is not None:
            return LlamaGrammar.from_string(options.grammar, verbose=False)
        return None

    @staticmethod
    def _trim_prompt(prompt: str) -> str:
        if len(prompt) < 800:
            return prompt
        return prompt[-(len(prompt) // 2):]

    def _generate_once(
        self,
        prompt: str,
        generation_options: GenerationOptions,
        max_tokens: int,
        stops: list[str],
    ) -> str:
        grammar = self._build_grammar(generation_options)
        temperature = (
            generation_options.temperature
            if generation_options.temperature is not None
            else 0.0
        )
        request_kwargs: dict[str, object] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "stop": stops,
        }
        if grammar is not None:
            request_kwargs["grammar"] = grammar

        if generation_options.messages and hasattr(self.client, "create_chat_completion"):
            chat_kwargs = dict(request_kwargs)
            response = self.client.create_chat_completion(
                messages=[
                    {"role": message.get("role", "user"), "content": message.get("content", "")}
                    for message in generation_options.messages
                ],
                **chat_kwargs,
            )
            text = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        else:
            response = self.client(prompt, **request_kwargs)
            text = response.get("choices", [{}])[0].get("text", "")
        return apply_stop_sequences(text.strip(), stops)

    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        lock = getattr(self, "_generate_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._generate_lock = lock
        with lock:
            generation_options = options or GenerationOptions()
            stops = list(generation_options.stop) if generation_options.stop else [
                "User:",
                "Triage Assistant:",
                "</question>",
                "\nJSON OUTPUT:",
                "\nJSON REPORT OUTPUT:",
            ]
            max_tokens = (
                generation_options.max_new_tokens
                if generation_options.max_new_tokens is not None
                else self.config.max_new_tokens
            )
            first_prompt = prompt
            attempts: list[tuple[str, int]] = [
                (first_prompt, max_tokens),
                (self._trim_prompt(first_prompt), max(32, max_tokens // 2)),
            ]
            last_error: Exception | None = None

            for attempt_prompt, attempt_tokens in attempts:
                try:
                    bounded_tokens = self._resolve_max_tokens(attempt_prompt, attempt_tokens)
                    return self._generate_once(
                        attempt_prompt,
                        generation_options,
                        bounded_tokens,
                        stops,
                    )
                except LLMContextBudgetExceededError as err:
                    last_error = err
                    continue
                except Exception as err:
                    message = str(err)
                    if self._is_context_overflow_error(message):
                        last_error = LLMContextBudgetExceededError(message)
                        continue
                    if self._is_decode_error(message):
                        last_error = LLMDecodeError(message)
                        continue
                    raise LLMGenerationError(message) from err

            if isinstance(last_error, LLMGenerationError):
                raise last_error
            raise LLMGenerationError("llama.cpp generation failed after retry.")
