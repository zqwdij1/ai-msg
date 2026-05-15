"""LangGraph StateGraph: start → ask_question → evaluate_answer → next_question_or_end."""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from interview.llm import get_interview_llm
from interview.parsing import (
    extract_json_block,
    heuristic_flags_from_cot_stage3,
    parse_interviewer_react,
    split_evaluator_cot_stages,
)
from interview.prompts import (
    FINAL_REPORT_SYSTEM,
    INTERVIEWER_SYSTEM,
    EVALUATOR_SYSTEM,
    build_evaluator_user_prompt,
    build_final_report_user_prompt,
    build_interviewer_user_prompt,
)
from interview.questions import pick_questions_for_role
from interview.state import EvaluatorCoTRecord, InterviewState, InterviewerReActRecord

LlmFactory = Literal["interviewer", "evaluator", "final"]


def _default_llm_factory(which: LlmFactory):
    _ = which
    return get_interview_llm()


def coordinator(state: InterviewState) -> Command:
    """Coordinator：按阶段选择 start / 评估链 / 结束（无冗余 Agent 对话）。"""
    if state.get("interview_finished"):
        return Command(goto=END)
    ans = state.get("candidate_answer")
    has_ans = bool(ans and str(ans).strip())
    if has_ans and state.get("interview_initialized"):
        return Command(goto="evaluate_answer")
    if not state.get("interview_initialized"):
        return Command(goto="start")
    return Command(goto=END)


def start(state: InterviewState) -> dict[str, Any]:
    cap = int(state.get("max_main_questions") or 3)
    role = state.get("role_stack") or "通用技术"
    main_qs = pick_questions_for_role(role, cap)
    if not main_qs:
        main_qs = ["请做一个简要的自我介绍，并说明你最擅长的技术栈。"]
    return {
        "interview_initialized": True,
        "interview_finished": False,
        "main_questions": main_qs,
        "main_question_index": 0,
        "follow_ups_for_current_main": 0,
        "current_main_question_text": main_qs[0],
        "coordinator_route": "first_main",
        "accumulated_scores": state.get("accumulated_scores") or [],
        "final_report": None,
        "transcript": [
            {
                "type": "system",
                "content": f"面试开始 session={state.get('session_id','')}",
                "role_stack": role,
            }
        ],
    }


def ask_question(state: InterviewState, *, llm_factory=_default_llm_factory) -> dict[str, Any]:
    llm = llm_factory("interviewer")
    route = state.get("coordinator_route") or "first_main"
    summary = {
        "coordinator_route": route,
        "main_question_index": state.get("main_question_index"),
        "current_main_question_text": state.get("current_main_question_text"),
        "last_question_asked": state.get("last_question_asked"),
        "follow_ups_for_current_main": state.get("follow_ups_for_current_main"),
        "latest_evaluator_cot": state.get("latest_evaluator_cot"),
        "main_questions_total": len(state.get("main_questions") or []),
    }
    user = build_interviewer_user_prompt(json.dumps(summary, ensure_ascii=False, indent=2))
    msg = llm.invoke(
        [
            SystemMessage(content=INTERVIEWER_SYSTEM),
            HumanMessage(content=user),
        ]
    )
    raw = getattr(msg, "content", str(msg))
    parsed = parse_interviewer_react(str(raw))
    utterance = str(parsed.get("utterance") or "").strip() or str(state.get("current_main_question_text") or "")
    react: InterviewerReActRecord = {
        "thought": str(parsed.get("thought") or ""),
        "action_type": parsed.get("action_type", "FOLLOW_UP"),
        "action_detail": str(parsed.get("utterance") or ""),
    }
    entry = {
        "type": "interviewer",
        "route": route,
        "thought": react["thought"],
        "action_type": react["action_type"],
        "question": utterance,
    }
    return {
        "last_question_asked": utterance,
        "latest_interviewer_react": react,
        "transcript": [entry],
    }


def evaluate_answer(state: InterviewState, *, llm_factory=_default_llm_factory) -> dict[str, Any]:
    llm = llm_factory("evaluator")
    question = state.get("last_question_asked") or ""
    answer = (state.get("candidate_answer") or "").strip()
    role = state.get("role_stack") or "通用技术"
    user = build_evaluator_user_prompt(question=question, answer=answer, role_stack=role)
    msg = llm.invoke(
        [
            SystemMessage(content=EVALUATOR_SYSTEM),
            HumanMessage(content=user),
        ]
    )
    raw = str(getattr(msg, "content", str(msg)))
    s1, s2, s3 = split_evaluator_cot_stages(raw)
    data = extract_json_block(raw) or {}
    rel = str(data.get("relevance_label") or "部分相关")
    if rel not in {"相关", "部分相关", "不相关"}:
        rel = "部分相关"
    score = float(data.get("score") if data.get("score") is not None else 5.0)
    score = max(0.0, min(10.0, score))
    w_fu = float(data.get("weight_follow_up", 0.5))
    w_nq = float(data.get("weight_next_question", 0.5))
    w_fu = max(0.0, min(1.0, w_fu))
    w_nq = max(0.0, min(1.0, w_nq))
    h_fu, h_complete = heuristic_flags_from_cot_stage3(s3)
    should_follow_up = h_fu or (w_fu >= 0.55 and w_fu > w_nq and rel != "不相关")
    question_complete = h_complete or (w_nq >= 0.55 and w_nq > w_fu) or rel == "不相关"
    improvement = str(data.get("improvement_suggestions") or "")
    cot: EvaluatorCoTRecord = {
        "stage1_relevance": s1,
        "stage2_coverage": s2,
        "stage3_score_and_advice": s3,
        "relevance_label": rel,  # type: ignore[assignment]
        "score": score,
        "improvement_suggestions": improvement,
        "weight_follow_up": w_fu,
        "weight_next_question": w_nq,
    }
    prev_scores = list(state.get("accumulated_scores") or [])
    cand_entry = {
        "type": "candidate",
        "answer": answer,
        "for_question": question,
    }
    eval_entry = {
        "type": "evaluator",
        "cot": cot,
        "raw_cot": raw,
        "should_follow_up": should_follow_up,
        "question_complete": question_complete,
    }
    return {
        "latest_evaluator_cot": cot,
        "should_follow_up": should_follow_up,
        "question_complete": question_complete,
        "accumulated_scores": [*prev_scores, score],
        "transcript": [cand_entry, eval_entry],
        "candidate_answer": None,
    }


def next_question_or_end(state: InterviewState) -> Command:
    L = state.get("main_questions") or []
    cap = min(len(L), int(state.get("max_main_questions") or len(L) or 1))
    idx = int(state.get("main_question_index") or 0)
    fu = int(state.get("follow_ups_for_current_main") or 0)
    max_fu = int(state.get("max_follow_ups_per_main") or 2)
    evalr = state.get("latest_evaluator_cot") or {}
    w_fu = float(evalr.get("weight_follow_up", 0.5))
    w_nq = float(evalr.get("weight_next_question", 0.5))
    should_fu = bool(state.get("should_follow_up"))
    complete = bool(state.get("question_complete"))
    if fu >= max_fu:
        complete = True
    prefer_follow = should_fu and not complete and fu < max_fu and w_fu >= w_nq
    if prefer_follow:
        return Command(
            goto="ask_question",
            update={
                "follow_ups_for_current_main": fu + 1,
                "coordinator_route": "follow_up",
            },
        )
    nidx = idx + 1
    if nidx >= cap:
        return Command(goto="final_report", update={"coordinator_route": "final_report"})
    return Command(
        goto="ask_question",
        update={
            "main_question_index": nidx,
            "follow_ups_for_current_main": 0,
            "current_main_question_text": L[nidx],
            "coordinator_route": "next_main",
        },
    )


def final_report(state: InterviewState, *, llm_factory=_default_llm_factory) -> dict[str, Any]:
    llm = llm_factory("final")
    lines: list[str] = []
    for row in state.get("transcript") or []:
        if row.get("type") == "interviewer":
            lines.append(f"面试官：{row.get('question')}")
        elif row.get("type") == "candidate":
            lines.append(f"候选人：{row.get('answer')}")
    text = "\n".join(lines)
    scores = list(state.get("accumulated_scores") or [])
    user = build_final_report_user_prompt(text, scores)
    msg = llm.invoke(
        [
            SystemMessage(content=FINAL_REPORT_SYSTEM),
            HumanMessage(content=user),
        ]
    )
    report = str(getattr(msg, "content", str(msg))).strip()
    return {
        "final_report": report,
        "interview_finished": True,
        "transcript": [{"type": "final_report", "content": report}],
    }


def build_graph(*, llm_factory=_default_llm_factory) -> StateGraph:
    g = StateGraph(InterviewState)

    def _ask(s: InterviewState) -> dict[str, Any]:
        return ask_question(s, llm_factory=llm_factory)

    def _eval(s: InterviewState) -> dict[str, Any]:
        return evaluate_answer(s, llm_factory=llm_factory)

    def _final(s: InterviewState) -> dict[str, Any]:
        return final_report(s, llm_factory=llm_factory)

    g.add_node("coordinator", coordinator)
    g.add_node("start", start)
    g.add_node("ask_question", _ask)
    g.add_node("evaluate_answer", _eval)
    g.add_node("next_question_or_end", next_question_or_end)
    g.add_node("final_report", _final)

    g.add_edge(START, "coordinator")
    g.add_edge("start", "ask_question")
    g.add_edge("ask_question", END)
    g.add_edge("evaluate_answer", "next_question_or_end")
    g.add_edge("final_report", END)
    return g


def build_interview_app(*, llm_factory=_default_llm_factory):
    """Compiled app with in-memory checkpointer (thread_id = session_id)."""
    checkpointer = MemorySaver()
    graph = build_graph(llm_factory=llm_factory)
    return graph.compile(checkpointer=checkpointer)


def fresh_session_id() -> str:
    return str(uuid.uuid4())
