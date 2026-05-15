"""简单 Web 演示（Streamlit）。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import streamlit as st

from interview.graph import build_interview_app, fresh_session_id
from interview.llm import has_llm_credentials


def get_app():
    if "compiled_app" not in st.session_state:
        st.session_state.compiled_app = build_interview_app()
    return st.session_state.compiled_app


def main():
    st.set_page_config(page_title="多 Agent 技术面试", layout="wide")
    st.title("基于 LangGraph 的多 Agent 技术面试")

    if not has_llm_credentials():
        st.error("请配置 OPENAI_API_KEY 或 ZHIPU_API_KEY（环境变量或项目根目录 .env）。")
        return

    with st.sidebar:
        st.subheader("会话")
        if st.button("新面试"):
            new_id = fresh_session_id()
            st.session_state.clear()
            st.session_state.sid = new_id
            st.rerun()
        role = st.text_input("技术方向", value="后端 / 数据库")
        max_main = st.number_input("主问题数上限", min_value=1, max_value=5, value=2)
        max_fu = st.number_input("每题追问上限", min_value=0, max_value=4, value=2)

    app = get_app()
    if "sid" not in st.session_state:
        st.session_state.sid = fresh_session_id()

    cfg = {"configurable": {"thread_id": st.session_state.sid}}
    st.caption(f"session_id = `{st.session_state.sid}`")

    if "bootstrapped" not in st.session_state:
        st.session_state.state = app.invoke(
            {
                "session_id": st.session_state.sid,
                "role_stack": role,
                "max_main_questions": int(max_main),
                "max_follow_ups_per_main": int(max_fu),
            },
            cfg,
        )
        st.session_state.bootstrapped = True

    state = st.session_state.state

    if state.get("interview_finished") and state.get("final_report"):
        st.success("面试已结束")
        st.markdown(state["final_report"])
        if st.button("查看评估推理（最后一轮）"):
            st.json(state.get("latest_evaluator_cot") or {})
        return

    st.markdown("### 当前问题")
    st.info(state.get("last_question_asked") or "")

    ev = state.get("latest_evaluator_cot")
    if ev:
        with st.expander("上一轮 Evaluator CoT（可审计）"):
            st.write("阶段1 相关性：", ev.get("stage1_relevance", ""))
            st.write("阶段2 覆盖：", ev.get("stage2_coverage", ""))
            st.write("阶段3 评分与建议：", ev.get("stage3_score_and_advice", ""))
            st.json(
                {
                    "score": ev.get("score"),
                    "weight_follow_up": ev.get("weight_follow_up"),
                    "weight_next_question": ev.get("weight_next_question"),
                    "relevance_label": ev.get("relevance_label"),
                }
            )

    ir = state.get("latest_interviewer_react")
    if ir:
        with st.expander("当前轮 Interviewer ReAct"):
            st.write("Thought:", ir.get("thought", ""))
            st.write("Action:", ir.get("action_type", ""))

    ans = st.text_area("你的回答", height=160, key="ans_box")
    if st.button("提交回答"):
        if not ans.strip():
            st.warning("请输入回答。")
        else:
            st.session_state.state = app.invoke({"candidate_answer": ans.strip()}, cfg)
            st.rerun()


if __name__ == "__main__":
    main()
