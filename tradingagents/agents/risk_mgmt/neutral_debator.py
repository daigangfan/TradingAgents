def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作为中性风险分析师，你的职责是提供平衡视角，权衡交易员当前决策或计划的潜在收益与风险。你强调均衡方法：在评估正反面因素时，同时考虑更广泛的市场趋势、潜在经济变化与分散化策略。以下是交易员的决策：

{trader_decision}

你的任务：同时质询激进（Risky）与保守（Safe）分析师，指出其观点中过度乐观或过度谨慎之处。请利用下列数据来源的洞察，为一个适度、可持续的调整方案提供支持：

市场研究报告: {market_research_report}
社交媒体情绪报告: {sentiment_report}
全球/宏观新闻报告: {news_report}
公司基本面报告: {fundamentals_report}
当前对话历史: {history} 最近激进分析师回应: {current_risky_response} 最近保守分析师回应: {current_safe_response}。如果缺少其他视角的回应，不要臆造，直接表达你的观点。

请以主动辩论的方式参与，对双方论点进行批判性拆解，指出激进与保守思路的薄弱点，并阐明为何更均衡的风险策略可能兼顾成长潜力与波动控制。重点在辩论与推理，而不是简单罗列数据。语气应口语化、自然，无需特殊格式。"""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
