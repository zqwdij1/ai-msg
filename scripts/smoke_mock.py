from types import SimpleNamespace

from langgraph.checkpoint.memory import MemorySaver

from interview.graph import build_graph


def mock_factory(which):
    class Mock:
        def invoke(self, messages):
            if which == "interviewer":
                return SimpleNamespace(
                    content="Thought: ok\nAction: FOLLOW_UP\nUtterance: Q?"
                )
            if which == "evaluator":
                return SimpleNamespace(
                    content=(
                        "阶段1\n相关\n阶段2\nx\n阶段3\n建议追问：否\n主问题完成度：已达标\n"
                        '```json\n{"relevance_label":"相关","score":8,'
                        '"improvement_suggestions":"","weight_follow_up":0.1,'
                        '"weight_next_question":0.9}\n```'
                    )
                )
            return SimpleNamespace(content="REPORT")

    return Mock()


def main():
    app = build_graph(llm_factory=mock_factory).compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "mock-1"}}
    s1 = app.invoke(
        {"session_id": "m1", "max_main_questions": 2, "max_follow_ups_per_main": 1},
        cfg,
    )
    print("1", s1.get("interview_initialized"), s1.get("last_question_asked"))
    s2 = app.invoke({"candidate_answer": "ans"}, cfg)
    print("2", s2.get("interview_finished"), s2.get("main_question_index"), s2.get("coordinator_route"))
    print("rep", s2.get("final_report"))


if __name__ == "__main__":
    main()
