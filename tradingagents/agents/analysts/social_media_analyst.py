from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_social_media_analyst(llm, toolkit):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_stock_news_openai]
        else:
            tools = [
                toolkit.get_reddit_stock_info,
            ]

        system_message = (
            "你是一名社交媒体与公司专项资讯研究/分析助手，任务是分析过去一周中与某特定公司相关的社交媒体帖子、最新公司新闻及公众情绪。系统会提供公司名称，你需撰写一份深入全面的长篇报告：包括你的分析过程、洞察发现、以及对交易员和投资者可能的影响。请综合多来源：社交媒体讨论、逐日情绪数据、公司事件与新闻。不仅要总结事实，还要抽取结构化逻辑与潜在风险/机会。不要简单说‘趋势复杂/混合’，必须输出细致入微、支持决策的分析。"
            + """ 请在报告末尾附上一张 Markdown 表格，总结关键要点，结构清晰便于阅读。""",
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一名协作型 AI 助手。"
                    " 请使用可用工具推进分析。"
                    " 若暂无法完整回答，可由其它具不同工具的助手继续。"
                    " 先执行你当前能完成的部分以推动进展。"
                    " 若你或其它助手已获得 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**（最终交易提案），"
                    " 请在回复最前添加 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** 提醒团队停止。"
                    " 你可以使用这些工具：{tool_names}.\n{system_message}"
                    " 当前日期：{current_date}。我们正在分析的公司：{ticker}。",
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
