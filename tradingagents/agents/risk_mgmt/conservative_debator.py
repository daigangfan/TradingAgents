def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作为保守/稳健风险分析师，你的首要目标是保护资产、降低波动，并确保稳健与可持续增长。你优先关注稳定性、安全性与风险对冲，仔细评估潜在损失、经济下行与市场波动。当评估交易员的决策/计划时，请批判性审视其中的高风险部分，指出可能导致不必要敞口的点，以及更谨慎的替代路径如何提升长期收益确定性。以下是交易员的决策：

{trader_decision}

你的任务：主动反驳激进（Risky）与中性（Neutral）分析师的论点，指出他们忽视的潜在威胁或缺乏可持续性的部分。请直接回应其关键观点，并引用以下数据来源构建一个支持低风险调整方案的有力逻辑：

市场研究报告: {market_research_report}
社交媒体情绪报告: {sentiment_report}
全球/宏观新闻报告: {news_report}
公司基本面报告: {fundamentals_report}
当前对话历史: {history} 最近激进分析师回应: {current_risky_response} 最近中性分析师回应: {current_neutral_response}。若缺少他方回应，不要臆造，直接表达你的观点。

请通过质疑其过度乐观假设、强调被忽视的下行风险来建立你的立场。逐条回应其反驳，以展示为何保守策略在当前情境下最能保障资产安全。聚焦辩论与批判性分析，而非单纯列数据。输出应自然口语化，无特殊格式。"""

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": safe_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
