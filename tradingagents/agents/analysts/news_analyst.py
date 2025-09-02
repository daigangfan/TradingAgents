from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_global_news_openai]
        else:
            tools = [
                toolkit.get_finnhub_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        system_message = (
            "你是一名新闻研究员，任务是分析过去一周的最新新闻与趋势。请撰写一份与交易及宏观经济相关的全面报告，尽可能覆盖全球与市场层面动态。需综合 EODHD 与 finnhub 等来源。不要只说‘趋势是混合的’，而要提供细致、结构化、具操作价值的洞察，帮助交易员做出决策。"
            + """ 请在报告末尾附上一张 Markdown 表格，总结关键要点，条理清晰、便于阅读。"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一名协同工作的 AI 助手。"
                    " 请使用提供的工具推进问题解答。"
                    " 若暂时无法完整回答，可由其他具不同工具的助手继续。"
                    " 尽你所能先完成可推进的步骤。"
                    " 如果你或其他助手已形成 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**（最终交易提案），"
                    " 请在回复开头加上 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** 以提示团队停止进一步分析。"
                    " 你可使用以下工具：{tool_names}.\n{system_message}"
                    " 当前日期：{current_date}。我们关注的公司：{ticker}。",
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
            "news_report": report,
        }

    return news_analyst_node
