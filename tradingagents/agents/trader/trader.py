import functools


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for rec in past_memories:
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"基于分析团队的综合研究，下方为针对 {company_name} 制定的投资计划。该计划融合了当前技术面趋势、宏观经济信号与社交媒体情绪洞察。请以此作为你下一步交易决策的基础。\n\n拟议投资计划: {investment_plan}\n\n请利用这些洞察，做出信息充分且具有策略性的判断。",
        }

        messages = [
            {
                "role": "system",
                "content": f"""你是一名交易代理，负责基于市场数据做出投资判断。请在分析后给出明确的 买入 / 卖出 / 持有 建议，并在结尾使用 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'（按实际选择替换）以确认你的结论。务必参考过往相似情境的反思以避免重复错误。以下是相似情境中的反思与经验教训：{past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
