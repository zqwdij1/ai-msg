"""Parse LLM outputs for Interviewer (ReAct) and Evaluator (CoT + JSON)."""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def parse_interviewer_react(text: str) -> dict[str, Any]:
    thought_m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.S | re.I)
    action_m = re.search(r"Action:\s*([A-Z_]+)", text, re.I)
    utter_m = re.search(r"Utterance:\s*(.+)\Z", text, re.S | re.I)
    thought = (thought_m.group(1).strip() if thought_m else "").strip()
    action_raw = (action_m.group(1).upper() if action_m else "FOLLOW_UP").strip()
    utterance = (utter_m.group(1).strip() if utter_m else text.strip()).strip()
    if action_raw not in {"FOLLOW_UP", "NEXT_MAIN_QUESTION", "END_INTERVIEW"}:
        action_raw = "FOLLOW_UP"
    return {
        "thought": thought,
        "action_type": action_raw,
        "utterance": utterance,
    }


def extract_json_block(text: str) -> Optional[dict[str, Any]]:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = fence.group(1) if fence else None
    if not raw:
        brace = re.search(r"\{[\s\S]*\}\s*$", text.strip())
        raw = brace.group(0) if brace else None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def split_evaluator_cot_stages(text: str) -> tuple[str, str, str]:
    """Best-effort split for audit fields; falls back to full text."""
    s1 = re.search(r"阶段\s*1[^\n]*\n([\s\S]+?)(?=阶段\s*2|\Z)", text)
    s2 = re.search(r"阶段\s*2[^\n]*\n([\s\S]+?)(?=阶段\s*3|\Z)", text)
    s3 = re.search(r"阶段\s*3[^\n]*\n([\s\S]+?)(?=```json|\Z)", text)
    a = s1.group(1).strip() if s1 else text
    b = s2.group(1).strip() if s2 else ""
    c = s3.group(1).strip() if s3 else ""
    return a, b, c


def heuristic_flags_from_cot_stage3(stage3: str) -> tuple[bool, bool]:
    """Parse Chinese hints for follow-up / main complete."""
    fu = "建议追问：是" in stage3 or "建议追问:是" in stage3
    complete = "主问题完成度：已达标" in stage3 or "主问题完成度:已达标" in stage3
    return fu, complete
