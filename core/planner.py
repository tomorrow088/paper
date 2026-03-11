"""
规划 Agent：负责分析用户意图，决定调用哪些工具
这是整个 Agent 系统的"大脑"
"""

from zai import ZhipuAiClient
from config import ZHIPU_API_KEY, LLM_MODEL, LLM_TEMPERATURE


class PlannerAgent:
    """
    规划Agent：分析用户意图，输出执行计划
    支持的意图：
      - summarize:   总结论文/章节
      - qa:          基于论文的问答
      - compare:     对比分析（章节间/概念间）
      - extract:     提取关键信息（方法、公式、实验结果等）
    """

    def __init__(self):
        self.client = ZhipuAiClient(api_key=ZHIPU_API_KEY)

        # 定义可用的工具描述（供大模型理解）
        self.tools_description = """
            你有以下工具可以调用：
            1. [retrieve] 检索工具：从论文向量库中检索与问题相关的文本片段。适用于需要查找具体内容的场景。
            2. [summarize] 总结工具：对检索到的内容进行归纳总结。适用于用户想要概览、摘要的场景。
            3. [qa] 问答工具：基于检索内容严格回答用户问题。适用于用户提出具体学术问题的场景。
            4. [compare] 对比工具：对不同章节或概念进行对比分析。适用于用户要求比较、分析异同的场景。
            5. [extract] 提取工具：从论文中提取结构化信息（如方法步骤、实验数据、公式）。适用于用户要求列出、提取具体内容的场景。
            """

    def plan(self, question: str, chat_history: list = None) -> dict:
        """
        分析用户意图，返回执行计划
        返回格式: {"intent": "qa", "tools": ["retrieve", "qa"], "rewritten_query": "..."}
        """
        history_text = ""
        if chat_history:
            # 只取最近 3 轮对话作为上下文
            recent = chat_history[-6:]
            history_text = "\n".join(
                [f"{'用户' if m['role']=='user' else '助手'}: {m['content'][:200]}" for m in recent]
            )

        prompt = f"""你是一个学术论文助手的规划器。请分析用户的意图，并输出一个执行计划。

            {self.tools_description}
            
            对话历史：
            {history_text if history_text else "（无）"}
            
            用户最新问题：{question}
            
            请严格按以下 JSON 格式输出（不要输出其他内容）：
            {{
                "intent": "意图类型（summarize/qa/compare/extract）",
                "tools": ["需要依次调用的工具列表"],
                "rewritten_query": "改写后的更专业的检索查询语句",
                "reasoning": "一句话解释你的规划理由"
            }}
            """

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个JSON输出器，只输出合法的JSON，不输出任何其他文字。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # 规划需要确定性
        )

        raw = response.choices[0].message.content.strip()

        # 解析 JSON，容错处理
        import json
        try:
            # 去除可能的 markdown 代码块标记
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            plan = json.loads(raw)
        except json.JSONDecodeError:
            # 如果解析失败，回退到默认的 qa 流程
            plan = {
                "intent": "qa",
                "tools": ["retrieve", "qa"],
                "rewritten_query": question,
                "reasoning": "规划解析失败，使用默认问答流程"
            }

        # 确保必要字段存在
        plan.setdefault("intent", "qa")
        plan.setdefault("tools", ["retrieve", "qa"])
        plan.setdefault("rewritten_query", question)

        return plan
