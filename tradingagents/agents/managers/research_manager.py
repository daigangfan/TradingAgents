def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for _, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""作为投资组合经理与辩论协调者，你需要对本轮辩论进行批判性评估，并做出明确决策：支持看空（Bear）、看多（Bull），或在有充分论证时选择 Hold（而非模糊折中）。

请简洁总结双方最有力的核心论据，聚焦最具说服力的证据与推理。你的建议（Buy / Sell / Hold）必须清晰、可执行。不要因为双方都“有道理”而默认 Hold，应基于最强论据坚定选择。

此外，请给出一份详细的交易/投资计划，包括：
1. 建议（Recommendation）：明确立场与其支撑论据。
2. 理由（Rationale）：解释为何这些观点足以得出结论。
3. 策略行动（Strategic Actions）：执行该建议的具体步骤。
请结合相似情境的过往失误，利用这些反思优化你的决策逻辑，体现学习改进。输出应自然对话风格，无需特殊格式。

以下是你的历史反思：
"{past_memory_str}"

以下为当前辩论内容：
Debate History:
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
