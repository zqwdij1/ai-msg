"""Configurable chat model for swapping OpenAI-compatible endpoints."""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


def has_llm_credentials() -> bool:
    """是否已配置任一可用的 API Key（OpenAI 或智谱）。"""
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ZHIPU_API_KEY"))


def get_interview_llm(
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> BaseChatModel:
    """
    Build the LLM used by Interviewer / Evaluator.

    Environment:
      OPENAI_API_KEY — 或填智谱 Key（与 ZHIPU_API_KEY 二选一）
      ZHIPU_API_KEY — 智谱 Key；若设置且未设 OPENAI_API_KEY 则自动使用
      OPENAI_BASE_URL — 智谱示例：https://open.bigmodel.cn/api/paas/v4/
      INTERVIEW_MODEL — 智谱示例：glm-4-flash；OpenAI 默认：gpt-4o-mini
    """
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
    if not base_url and os.getenv("ZHIPU_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        base_url = "https://open.bigmodel.cn/api/paas/v4/"
    is_zhipu = bool(base_url and "bigmodel.cn" in base_url)
    default_model = "glm-4-flash" if is_zhipu else "gpt-4o-mini"
    resolved = model or os.getenv("INTERVIEW_MODEL", default_model)
    return ChatOpenAI(
        model=resolved,
        temperature=temperature,
        timeout=timeout,
        base_url=base_url,
        api_key=api_key,
    )
