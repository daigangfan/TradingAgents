def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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

        prompt = f"""你是一名看多（Bull）分析师，主张对该股票进行投资。你的任务是构建一个有力且以证据为基础的论证，突出其增长潜力、竞争优势与积极市场信号。请利用提供的研究与数据回应质疑，并有效反驳看空（Bear）观点。

需要重点关注：
- 增长潜力：市场空间、收入预期、可扩展性。
- 竞争优势：产品独特性、品牌力、市场主导位置等。
- 积极指标：财务健康度、行业趋势、近期正面新闻。
- 针对 Bear 反驳：用具体数据与严谨推理拆解其担忧，说明为何多头观点更具说服力。
- 互动性：用对话式方式呈现，直接回应对方观点，而非生硬罗列表格数据。

可用资源：
市场研究报告: {market_research_report}
社交媒体情绪报告: {sentiment_report}
全球/宏观新闻: {news_report}
公司基本面报告: {fundamentals_report}
当前辩论历史: {history}
最新看空论点: {current_response}
相似情境的反思与经验教训: {past_memory_str}

请结合以上信息，输出一个令人信服的看多论证，反驳看空观点，并在动态辩论中展示多头立场的优势；同时体现你从历史反思中吸取的改进。"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
