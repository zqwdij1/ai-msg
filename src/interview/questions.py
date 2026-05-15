"""Static question bank for demo; admin can extend or replace."""

from __future__ import annotations

DEFAULT_MAIN_QUESTIONS: list[str] = [
    "请解释什么是索引覆盖（covering index）？它如何影响查询性能？",
    "简述 Redis 持久化 RDB 与 AOF 的差异及适用场景。",
    "在分布式系统中，如何理解 CAP 定理？请结合可用性与分区容错举例。",
]


def pick_questions_for_role(role_stack: str, max_n: int) -> list[str]:
    # Placeholder: future load from DB by role_stack
    _ = role_stack
    return DEFAULT_MAIN_QUESTIONS[:max_n]
