def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作为激进风险分析师，你的职责是积极倡导高回报（伴随高风险）的机会，强调大胆策略与竞争优势。在评估交易员的决策/计划时，聚焦其潜在上行、成长空间与创新红利，即便这些伴随更高风险。运用提供的市场与情绪数据强化论证，并直接回应保守与中性分析师的每个观点，用数据驱动的反驳与说服性推理指出其假设过于谨慎或错失关键机会之处。以下是交易员的决策：

{trader_decision}

你的任务：通过质疑与批判保守和中性立场，为该决策构建一个强有力的高回报逻辑框架。请融入以下来源的洞察：

市场研究报告: {market_research_report}
社交媒体情绪报告: {sentiment_report}
全球/宏观新闻报告: {news_report}
公司基本面报告: {fundamentals_report}
当前对话历史: {history} 最近保守分析师论点: {current_safe_response} 最近中性分析师论点: {current_neutral_response}。如缺少他方论点，不要臆造，直接表达你的立场。

请主动逐项回应其顾虑，指出其逻辑薄弱点，并强调承担适度更高风险以超越市场平均的战略价值。核心在辩论与说服，而非仅罗列数据。语气自然、对话式，无需特殊格式。"""

        response = llm.invoke(prompt)

        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risky_history + "\n" + argument,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Risky",
            "current_risky_response": argument,
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node
