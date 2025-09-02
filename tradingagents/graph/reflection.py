# TradingAgents/graph/reflection.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI


class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """Get the system prompt for reflection."""
        return """
你是一名资深金融分析师，负责回顾交易决策/分析，并给出全面、分步骤的反思性分析。
你的目标：输出对投资决策的细致洞察，指出改进空间。请严格遵循以下准则：

1. 推理（Reasoning）：
    - 针对每个交易决策，判断其为“正确”或“错误”：正确 = 带来收益提升；错误 = 造成收益下降/机会损失。
    - 分析每次成功或失误的成因，可考虑：
      - 市场情报（Market intelligence）
      - 技术指标（Technical indicators）
      - 技术信号（Technical signals）
      - 价格走势分析（Price movement analysis）
      - 整体市场数据分析（Overall market data analysis）
      - 新闻分析（News analysis）
      - 社交媒体与情绪分析（Social media & sentiment analysis）
      - 基本面数据分析（Fundamental analysis）
      - 对上述各因素在决策中的权重进行主观权衡并说明理由。

2. 改进（Improvement）：
    - 对“错误”或次优决策提出可提升收益的调整方案。
    - 给出明确的纠正/优化动作清单（例如：在某日期应将 HOLD 调整为 BUY，并说明触发条件）。

3. 总结（Summary）：
    - 汇总成功与失败带来的经验教训。
    - 指出这些经验在未来相似场景中的可迁移性，并建立情境之间的联系。

4. 精炼查询（Query）：
    - 将核心经验与推理压缩为一条不超过 1000 tokens 的精炼语句。
    - 确保该语句清晰捕捉教训要点与逻辑依据，便于快速检索与复用。

请严格遵循上述要求，输出须详实、准确、可执行。你还会得到客观的市场描述（价格走势、技术指标、新闻、情绪等），请结合这些上下文进行更具洞察力的分析。
"""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """Extract the current market situation from the state."""
        curr_market_report = current_state["market_report"]
        curr_sentiment_report = current_state["sentiment_report"]
        curr_news_report = current_state["news_report"]
        curr_fundamentals_report = current_state["fundamentals_report"]

        return f"{curr_market_report}\n\n{curr_sentiment_report}\n\n{curr_news_report}\n\n{curr_fundamentals_report}"

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses
    ) -> str:
        """Generate reflection for a component."""
        messages = [
            ("system", self.reflection_system_prompt),
            (
                "human",
                f"Returns: {returns_losses}\n\nAnalysis/Decision: {report}\n\nObjective Market Reports for Reference: {situation}",
            ),
        ]

        result = self.quick_thinking_llm.invoke(messages).content
        return result

    def reflect_bull_researcher(self, current_state, returns_losses, bull_memory):
        """Reflect on bull researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bull_debate_history = current_state["investment_debate_state"]["bull_history"]

        result = self._reflect_on_component(
            "BULL", bull_debate_history, situation, returns_losses
        )
        bull_memory.add_situations([(situation, result)])

    def reflect_bear_researcher(self, current_state, returns_losses, bear_memory):
        """Reflect on bear researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bear_debate_history = current_state["investment_debate_state"]["bear_history"]

        result = self._reflect_on_component(
            "BEAR", bear_debate_history, situation, returns_losses
        )
        bear_memory.add_situations([(situation, result)])

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        """Reflect on trader's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        trader_decision = current_state["trader_investment_plan"]

        result = self._reflect_on_component(
            "TRADER", trader_decision, situation, returns_losses
        )
        trader_memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, current_state, returns_losses, invest_judge_memory):
        """Reflect on investment judge's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["investment_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "INVEST JUDGE", judge_decision, situation, returns_losses
        )
        invest_judge_memory.add_situations([(situation, result)])

    def reflect_risk_manager(self, current_state, returns_losses, risk_manager_memory):
        """Reflect on risk manager's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["risk_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "RISK JUDGE", judge_decision, situation, returns_losses
        )
        risk_manager_memory.add_situations([(situation, result)])
