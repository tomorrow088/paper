"""
改进版生成器：根据不同意图使用不同的 Prompt 策略
每个工具对应一个专门优化的生成方法
"""

from zai import ZhipuAiClient
from config import ZHIPU_API_KEY, LLM_MODEL, LLM_TEMPERATURE


class ChatGenerator:

    def __init__(self):
        self.client = ZhipuAiClient(api_key=ZHIPU_API_KEY)

    # ==================== 工具方法 ====================

    def tool_qa(self, question: str, context_chunks: list, feedback: str = "", history: list = None) -> str:
        """问答工具：严格基于文献回答问题"""
        context_text = "\n\n".join(context_chunks)

        system_prompt = (
            "你是一个严谨的学术论文问答助手。\n"
            f"参考资料：\n{context_text}\n\n"
            "规则：\n"
            "1. 必须严格依据参考资料回答，不可编造。\n"
            "2. 如果参考资料中没有相关信息，回答：信息不足，无法回答。\n"
            "3. 回答时引用具体的论文内容作为依据。"
        )

        if feedback:
            system_prompt += f"\n\n【重要警告】上次回答未通过审查，审查意见：'{feedback}'。请根据此意见修正回答。"

        return self._call_llm(system_prompt, question, history)

    def tool_summarize(self, question: str, context_chunks: list, feedback: str = "", history: list = None) -> str:
        """总结工具：归纳论文内容"""
        context_text = "\n\n".join(context_chunks)

        system_prompt = (
            "你是一个学术论文总结专家。\n"
            f"参考资料：\n{context_text}\n\n"
            "规则：\n"
            "1. 用结构化的方式总结内容（背景、方法、结果、结论）。\n"
            "2. 突出论文的核心贡献和创新点。\n"
            "3. 使用学术但易懂的语言。\n"
            "4. 严格基于原文，不添加额外信息。"
        )

        if feedback:
            system_prompt += f"\n\n【修正要求】：'{feedback}'。"

        return self._call_llm(system_prompt, question, history)

    def tool_compare(self, question: str, context_chunks: list, feedback: str = "", history: list = None) -> str:
        """对比工具：分析不同概念/方法的异同"""
        context_text = "\n\n".join(context_chunks)

        system_prompt = (
            "你是一个学术对比分析专家。\n"
            f"参考资料：\n{context_text}\n\n"
            "规则：\n"
            "1. 明确列出对比维度（目标、方法、效果、适用场景等）。\n"
            "2. 使用表格或结构化格式展示对比结果。\n"
            "3. 给出你的分析结论。\n"
            "4. 所有对比内容必须有原文依据。"
        )

        if feedback:
            system_prompt += f"\n\n【修正要求】：'{feedback}'。"

        return self._call_llm(system_prompt, question, history)

    def tool_extract(self, question: str, context_chunks: list, feedback: str = "", history: list = None) -> str:
        """提取工具：提取结构化信息"""
        context_text = "\n\n".join(context_chunks)

        system_prompt = (
            "你是一个学术信息提取专家。\n"
            f"参考资料：\n{context_text}\n\n"
            "规则：\n"
            "1. 精确提取用户要求的信息（方法步骤、实验数据、公式、指标等）。\n"
            "2. 使用编号列表或表格呈现，结构清晰。\n"
            "3. 每条信息标注来源章节。\n"
            "4. 不做过度解读，忠实于原文表述。"
        )

        if feedback:
            system_prompt += f"\n\n【修正要求】：'{feedback}'。"

        return self._call_llm(system_prompt, question, history)

    # ==================== 验证方法 ====================

    def verify_answer(self, question: str, context_chunks: list, answer: str) -> str:
        """验证Agent：核查幻觉"""
        context_text = "\n\n".join(context_chunks)
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个极其严谨的学术同行评审员。\n"
                        "核查【生成的回答】是否完全基于【参考资料】。\n"
                        "规则：\n"
                        "1. 如果回答中有参考资料未提及的数据或臆断结论，输出修改意见。\n"
                        "2. 如果回答完全符合参考资料，仅输出：PASS\n"
                        "不要输出任何多余解释。"
                    )
                },
                {
                    "role": "user",
                    "content": f"【参考资料】\n{context_text}\n\n【用户问题】\n{question}\n\n【生成的回答】\n{answer}"
                }
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    # ==================== 内部方法 ====================

    def _call_llm(self, system_prompt: str, question: str, history: list = None) -> str:
        """统一的 LLM 调用入口"""
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": question})

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE
        )
        return response.choices[0].message.content

    # ==================== 工具分发器 ====================

    def dispatch_tool(self, tool_name: str, **kwargs) -> str:
        """
        根据工具名称分发到对应的方法
        这是 Agent 的核心——根据规划结果动态调用不同工具
        """
        tool_map = {
            "qa": self.tool_qa,
            "summarize": self.tool_summarize,
            "compare": self.tool_compare,
            "extract": self.tool_extract,
        }

        tool_func = tool_map.get(tool_name, self.tool_qa)  # 默认走 qa
        return tool_func(**kwargs)
