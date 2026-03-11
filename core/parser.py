import fitz     #PyMuPDF（常用 PDF 解析库
import os

class PaperParser:
    def __init__(self,file_path:str):       #构造函数，file_path: str 是类型标注，提醒“期望传字符串路径”。
        self.file_path = file_path
        if not os.path.exists((file_path)):
            raise FileNotFoundError(f"找不到文件：{file_path}")       #路径不对就立刻报错


    def extract_text(self) -> str:
        # """
        # 基础提取：将全篇PDF提取为纯文本字符串
        # """
        # full_text = ""
        # with pdfplumber.open(self.file_path) as pdf:
        #     for page in pdf.pages:
        #         page_text = page.extract_text()
        #         if page_text:
        #             full_text += page_text + "\n"

        """
        高级提取：按物理文本块（Block)提取内容，自动规避双栏错位拼接
        """
        doc = fitz.open(self.file_path)   #doc为提取到的pdf信息
        full_text = ""

        for page in doc:        #逐页处理
            blocks = page.get_text("blocks")        #拿“文本块”。每个 block 通常代表页面上一个连续的排版区域（对双栏论文更友好）

            text_blocks = [block[4] for block in blocks if block[6] == 0]

            full_text += "\n".join(text_blocks) + "\n"      #把同一页的文本块按换行拼起来，
        return full_text        #返回整篇纯文本
