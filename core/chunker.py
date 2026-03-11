import re       #正则表达式，用来找章节标题。
from langchain_text_splitters import RecursiveCharacterTextSplitter     #LangChain 的递归文本切分器（会按分隔符逐级尝试切块，直到满足 chunk_size）。



class PaperChunker:
    def __init__(self,chunk_size=1000,chunk_overlap=200):

        self.section_pattern = re.compile(
            r'(?:\d{1,2}\.?\s*)?(Abstract|Introduction|Methodology|Method|Related work|Experiment\s*results\s*and\s*analysis|Prospect|Subsequent\s*technology\s*research|Conclusions)\b\s*$',
            re.IGNORECASE | re.MULTILINE
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators = ["\n\n","\n","。",".",",","，"]      #优先按段落切，再按行切，再按句号/逗号切
        )

    def split_by_section(self, text: str) -> dict:
        """
        按章节结构切分长文本
        返回格式: {"Introduction": "...", "Method": "..."}
        """
        # 找出所有匹配章节标题的位置
        matches = list(self.section_pattern.finditer(text))

        if not matches:
            # 如果没匹配到标准章节，说明可能不是标准学术论文，直接返回整段
            return {"Full_Text": text}

        sections = {}
        # 遍历所有找到的标题，截取两个标题之间的文本内容
        for i, match in enumerate(matches):
            # 当前章节的名字（提取核心词汇并统一首字母大写）
            section_name = match.group(1).capitalize()
            # 当前章节内容的起点
            start_idx = match.end()
            # 当前章节内容的终点（如果是最后一个标题，则截取到文章末尾；否则截取到下一个标题前）
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # 提取文本并去除首尾的空白字符
            content = text[start_idx:end_idx].strip()
            # 过滤掉内容过短的无效章节（例如仅有几个字的干扰项）
            if len(content) > 50:
                chunks = self.text_splitter.split_text(content)  #调用切分器，将章节长文本转化为Chunk字符串列表
                sections[section_name] = chunks       # 将切分好的列表存入字典

        return sections