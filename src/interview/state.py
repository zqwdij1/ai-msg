"""Interview graph state schema (TypedDict + Pydantic helpers)."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Annotated, Any, Literal, Optional, TypedDict
from typing_extensions import NotRequired


def _extend_list(left: list[Any], right: list[Any]) -> list[Any]:
    if not right:
        return left
    return [*left, *right]


class EvaluatorCoTRecord(TypedDict, total=False):
    """Auditable CoT steps for one evaluation turn."""

    stage1_relevance: str
    stage2_coverage: str
    stage3_score_and_advice: str
    relevance_label: Literal["相关", "部分相关", "不相关"]
    score: float
    improvement_suggestions: str
    weight_follow_up: float
    weight_next_question: float


class InterviewerReActRecord(TypedDict, total=False):
    thought: str
    action_type: Literal["FOLLOW_UP", "NEXT_MAIN_QUESTION", "END_INTERVIEW"]
    action_detail: str


class InterviewState(TypedDict, total=False):
    """Shared LangGraph state for Coordinator / Interviewer / Evaluator."""

    session_id: str
    role_stack: str
    interview_initialized: bool
    interview_finished: bool

    main_question_index: int
    main_questions: list[str]
    max_main_questions: int
    follow_ups_for_current_main: int
    max_follow_ups_per_main: int

    current_main_question_text: str
    last_question_asked: str
    candidate_answer: Optional[str]

    # Append-only audit log (questions + answers + internal notes)
    transcript: Annotated[list[dict[str, Any]], _extend_list]

    # Latest structured agent outputs (overwritten each turn)
    latest_evaluator_cot: Optional[EvaluatorCoTRecord]
    latest_interviewer_react: Optional[InterviewerReActRecord]

    should_follow_up: bool
    question_complete: bool

    accumulated_scores: list[float]
    final_report: Optional[str]

    # Coordinator routing hint for debugging
    coordinator_route: NotRequired[str]


class EvaluatorCoTRecordModel(BaseModel):
    stage1_relevance: str = ""
    stage2_coverage: str = ""
    stage3_score_and_advice: str = ""
    relevance_label: str = ""
    score: float = 0.0
    improvement_suggestions: str = ""
    weight_follow_up: float = Field(default=0.5, ge=0.0, le=1.0)
    weight_next_question: float = Field(default=0.5, ge=0.0, le=1.0)


class InterviewStateModel(BaseModel):
    session_id: str = ""
    role_stack: str = "后端 / 数据库"
    interview_initialized: bool = False
    interview_finished: bool = False
    main_question_index: int = 0
    main_questions: list[str] = Field(default_factory=list)
    max_main_questions: int = 3
    follow_ups_for_current_main: int = 0
    max_follow_ups_per_main: int = 2
    current_main_question_text: str = ""
    last_question_asked: str = ""
    candidate_answer: Optional[str] = None
    should_follow_up: bool = False
    question_complete: bool = False
    accumulated_scores: list[float] = Field(default_factory=list)
    final_report: Optional[str] = None
