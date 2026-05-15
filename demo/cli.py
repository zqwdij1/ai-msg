"""命令行演示：一轮完整面试（需配置 OPENAI_API_KEY 或 ZHIPU_API_KEY）。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 允许直接 python demo/cli.py 运行
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from interview.graph import build_interview_app, fresh_session_id
from interview.llm import has_llm_credentials


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph 技术面试 CLI")
    parser.add_argument("--session", default="", help="恢复用的 session_id（默认新建）")
    parser.add_argument("--role", default="后端 / 数据库", help="职位技术方向描述")
    parser.add_argument("--max-main", type=int, default=2, help="主问题数量上限")
    parser.add_argument("--max-followup", type=int, default=2, help="每道主问题最大追问次数")
    args = parser.parse_args()

    if not has_llm_credentials():
        print("未检测到 OPENAI_API_KEY 或 ZHIPU_API_KEY。请在 .env 或环境变量中配置后重试。")
        sys.exit(1)

    app = build_interview_app()
    sid = args.session.strip() or fresh_session_id()
    cfg = {"configurable": {"thread_id": sid}}
    print(f"session_id={sid}（同一进程内可用相同 session 从检查点恢复）\n")

    snap = app.get_state(cfg)
    raw_vals = getattr(snap, "values", None) if snap is not None else None
    vals = dict(raw_vals) if isinstance(raw_vals, dict) else {}
    if (
        vals.get("interview_initialized")
        and not vals.get("interview_finished")
        and vals.get("last_question_asked")
    ):
        state = vals
        print("已从检查点恢复未完成的面试。\n")
    else:
        state = app.invoke(
            {
                "session_id": sid,
                "role_stack": args.role,
                "max_main_questions": args.max_main,
                "max_follow_ups_per_main": args.max_followup,
            },
            cfg,
        )

    while not state.get("interview_finished"):
        q = state.get("last_question_asked") or ""
        print("【面试官】", q, "\n", sep="")
        cot = state.get("latest_evaluator_cot")
        if cot:
            wfu = float(cot.get("weight_follow_up") or 0)
            wnq = float(cot.get("weight_next_question") or 0)
            print(
                f"（上一轮评估 分数={cot.get('score')} 追问权重={wfu:.2f} "
                f"下一题权重={wnq:.2f}）\n"
            )
        try:
            ans = input("【候选人】 ").strip()
        except EOFError:
            print("\n结束。")
            break
        if not ans:
            print("回答为空，请重新输入。")
            continue
        state = app.invoke({"candidate_answer": ans}, cfg)

    if state.get("final_report"):
        print("\n======== 最终评估 ========\n")
        print(state["final_report"])


if __name__ == "__main__":
    main()
