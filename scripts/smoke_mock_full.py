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
                        '"weight_follow_up":0.1,"weight_next_question":0.9}\n```'
                    )
                )
            return SimpleNamespace(content="FINAL_OK")

    return Mock()


app = build_graph(llm_factory=mock_factory).compile(checkpointer=MemorySaver())
cfg = {"configurable": {"thread_id": "t2"}}
app.invoke({"session_id": "t2", "max_main_questions": 2, "max_follow_ups_per_main": 1}, cfg)
app.invoke({"candidate_answer": "a1"}, cfg)
s = app.invoke({"candidate_answer": "a2"}, cfg)
print("finished", s.get("interview_finished"), "report", s.get("final_report"))
