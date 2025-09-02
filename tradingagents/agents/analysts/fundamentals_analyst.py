from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_fundamentals_analyst(llm, toolkit):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_fundamentals_openai]
        else:
            tools = [
                toolkit.get_finnhub_company_insider_sentiment,
                toolkit.get_finnhub_company_insider_transactions,
                toolkit.get_simfin_balance_sheet,
                toolkit.get_simfin_cashflow,
                toolkit.get_simfin_income_stmt,
            ]

        system_message = (
            "你是一名研究员，任务是分析某家公司在过去一周内的基础面信息。请撰写一份全面的报告，涵盖该公司的基础面要素：财务报表、公司概况、基础财务指标、历史财务表现、内部人士情绪以及内部人士交易等，以便为交易员提供完整视图。务必尽可能详尽，不要简单地说‘趋势是混合的’，而要提供细致入微的分析与有助于决策的洞察。请在报告末尾附上一张 Markdown 表格，总结关键要点，结构清晰、便于阅读。",
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一名有用的 AI 助手，正与其他助手协同工作。"
                    " 请使用提供的工具持续推进问题的解答。"
                    " 如果你无法完全回答也没关系；拥有不同工具的其他助手会在你停下的地方继续。"
                    " 请完成你力所能及的部分以推动进展。"
                    " 如果你或任何其他助手已经得出了 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**（最终交易提案：买入/持有/卖出）或最终交付内容，"
                    " 请在回复开头加上 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** 以便团队知道可以停止。"
                    " 你可以使用以下工具：{tool_names}.\n{system_message}"
                    " 供你参考，当前日期为 {current_date}。我们关注的公司/股票是 {ticker}。",
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
