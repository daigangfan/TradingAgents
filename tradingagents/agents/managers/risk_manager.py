import time
import json
def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:
        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for _, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""作为风险管理裁决者与辩论引导者，你需要评估三类风险分析师（激进 Risky / 中性 Neutral / 保守 Safe）的辩论内容，并为交易员确定最优行动路径。你的结论必须在：Buy / Sell / Hold 之间做出明确选择。只有在有充分、具体理由时才选择 Hold，而不是在观点分散时的默认折中。追求清晰与决断。

决策指引：
1. 关键要点总结：提炼每位分析师最有力、与情境最相关的论据。
2. 给出支撑理由：用辩论中的直接观点、引述与反驳支撑你的结论。
3. 优化交易员原计划：基于最初计划 **{trader_plan}**，结合分析师洞察进行必要调整。
4. 吸取历史经验：参考 **{past_memory_str}** 中的教训，避免重复错误，防止因错误的 BUY/SELL/HOLD 判断造成损失。

交付内容：
- 一个清晰、可执行的建议：Buy / Sell / Hold。
- 结合辩论与历史反思的详细理由。

---

辩论历史（Analysts Debate History）:
{history}

---

聚焦可执行洞察与持续改进。建立在历史经验之上，批判性整合多方观点，确保本次决策质量提升。"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
