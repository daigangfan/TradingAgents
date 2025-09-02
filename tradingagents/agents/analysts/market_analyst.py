from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_market_analyst(llm, toolkit):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_YFin_data_online,
                toolkit.get_stockstats_indicators_report_online,
            ]
        else:
            tools = [
                toolkit.get_YFin_data,
                toolkit.get_stockstats_indicators_report,
            ]

        system_message = (
            """你是一名交易分析助手，任务是分析金融市场。从以下列表中为当前市场环境或交易策略挑选**最相关的技术指标**。目标：选择不超过 **8 个** 互补且不冗余的指标。分类及对应指标：

移动平均 (Moving Averages):
- close_50_sma: 50 日简单均线，中期趋势指标。用途：识别趋势方向并作为潜在动态支撑/阻力。提示：存在价格滞后；与更快指标结合以改进入场时机。
- close_200_sma: 200 日简单均线，长期趋势基准。用途：确认总体市场方向、观察金叉/死叉。提示：反应最慢，更适合战略级趋势确认而非频繁交易。
- close_10_ema: 10 日指数均线，短期灵敏指标。用途：捕捉动能快速切换与潜在入场点。提示：震荡市易受噪声干扰；搭配较长均线过滤假信号。

MACD 相关 (MACD Related):
- macd: MACD 主线，通过不同 EMA 差值度量动能。用途：关注主线与信号线交叉及背离以判定趋势变化。提示：低波动或盘整需与其它确认工具联用。
- macds: MACD Signal 信号线，对 MACD 主线平滑。用途：与主线交叉可触发交易信号。提示：应嵌入更完整策略避免误触发。
- macdh: MACD Histogram 柱状图，展示主线与信号线差值。用途：可视化动能强弱并及早发现背离。提示：波动较大；快速行情下需辅以额外过滤。

动能指标 (Momentum):
- rsi: RSI 相对强弱指标，衡量动能识别超买/超卖。用途：经典 70/30 阈值及背离辅助反转判断。提示：强趋势中 RSI 可长时间极值；务必结合趋势结构验证。

波动率指标 (Volatility):
- boll: 布林中轨 (20SMA)。用途：作为价格波动的动态基线。提示：需与上/下轨联用识别突破或反转。
- boll_ub: 布林上轨，通常为中轨上方 2 倍标准差。用途：潜在超买或突破区域。提示：强趋势中价格可能“钉”在上轨。
- boll_lb: 布林下轨，通常为中轨下方 2 倍标准差。用途：潜在超卖区域。提示：需额外确认避免假反转。
- atr: ATR 平均真实波幅。用途：设定止损、按当前波动调整仓位。提示：属反应型指标，配合风险控制框架使用。

成交量类 (Volume-Based):
- vwma: 成交量加权移动平均。用途：融合价格与成交量验证趋势质量。提示：注意异常放量对加权的扭曲；与其它量能分析结合。

- 请选择信息互补的指标，避免冗余（例如不要同时选择 rsi 与概念上重复的其它同类动能指标）。并简述它们适用于当前市场情境的原因。调用工具时必须使用上面列出的精确指标名称（它们为已定义参数），否则调用会失败。请务必首先调用 get_YFin_data 以获取生成指标所需的 CSV。随后撰写一份非常详细、细腻的趋势分析报告。不要简单说“走势复杂/混合”，而要提供可支持交易决策的细粒度洞察。"""
            + """ 请在报告末尾附上一张 Markdown 表格，总结关键要点，结构清晰、便于阅读。"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一名有用的 AI 助手，正在与其他助手协作。"
                    " 请利用提供的工具推进问题求解。"
                    " 若你无法完整回答，不必勉强；拥有不同工具的助手会继续补充。"
                    " 请执行你力所能及的步骤来推动进展。"
                    " 如果你或任何助手已经生成 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**（最终交易提案），"
                    " 请在回复开头加入 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** 以提醒团队停止后续分析。"
                    " 你可以使用以下工具：{tool_names}.\n{system_message}"
                    " 供参考，当前日期：{current_date}。关注的公司/标的：{ticker}。",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content
       
        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
