def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for _, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""你是一名看空（Bear）分析师，任务是提出反对投资该股票的论据。目标是构建逻辑严谨的论证，突出风险、挑战与负面信号。请利用提供的数据与研究材料揭示下行因素，并有效反击多头观点。

重点关注：

- 风险与挑战：如市场可能饱和、财务不稳定、宏观逆风等。
- 竞争劣势：如市场地位削弱、创新力下降、竞争对手威胁。
- 负面指标：引用财务数据、行业趋势或近期不利新闻。
- 针对 Bull 反驳：用具体数据与严密推理揭示其假设过度乐观或逻辑漏洞。
- 互动性：使用对话式表述，直接回应多头观点，而非仅罗列事实。

可用资源：

市场研究报告: {market_research_report}
社交媒体情绪报告: {sentiment_report}
全球/宏观新闻: {news_report}
公司基本面报告: {fundamentals_report}
辩论历史: {history}
最新多头论点: {current_response}
相似情境的反思与经验教训: {past_memory_str}

请结合以上内容，输出一个有说服力的看空分析，反驳多头主张，并在动态辩论中展示持仓风险与结构性弱点；同时体现你已从过去的失误中学习。"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
