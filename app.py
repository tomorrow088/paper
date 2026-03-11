"""
Agent 版 PaperMind 后端
核心改动：加入 Planner Agent，让系统能自主决策执行路径

流程图：
  START
    │
    ▼
  [plan]  ← 规划Agent：分析意图，决定工具链
    │
    ▼
  [retrieve]  ← 检索Agent：根据改写后的查询检索文献
    │
    ▼
  [execute]  ← 执行Agent：根据规划结果，调用对应工具生成回答
    │
    ▼
  [verify]  ← 验证Agent：核查幻觉
    │
   ╱ ╲
  ╱   ╲
 PASS  FAIL ──→ [execute] (重写，最多3次)
  │         ╲
  ▼          → [error] (熔断)
 END
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from core.embedder import ZhipuEmbedder
from core.retriever import PaperRetriever
from core.generator import ChatGenerator
from core.planner import PlannerAgent
from core.parser import PaperParser
from core.chunker import PaperChunker
from config import ZHIPU_API_KEY
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import json
import uuid
import time
import random

memory = MemorySaver()

# ==================== State 定义 ====================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str           # 用户原始问题
    context: str            # 检索到的文献内容（chunk 列表）
    answer: str             # 生成的回答
    feedback: str           # 验证反馈
    retry_count: int        # 重试次数

    # ===== 新增：Agent 规划相关字段 =====
    plan: dict              # 规划Agent的输出（intent, tools, rewritten_query）
    current_tool: str       # 当前正在执行的工具名称


# ==================== 节点函数 ====================

def plan_node(state: AgentState):
    """
    🧭 规划节点（新增！）
    - 分析用户意图
    - 决定调用哪些工具
    - 改写查询语句
    """
    question = state["question"]
    chat_history = state.get("messages", [])

    print(f"\n🧭 [规划 Agent] 正在分析意图: '{question}'")

    plan = planner.plan(question, chat_history)

    print(f"   意图: {plan['intent']}")
    print(f"   工具链: {plan['tools']}")
    print(f"   改写查询: {plan.get('rewritten_query', question)}")
    print(f"   理由: {plan.get('reasoning', '')}")

    # 确定执行工具（取工具链中最后一个非 retrieve 的工具）
    exec_tools = [t for t in plan.get("tools", ["qa"]) if t != "retrieve"]
    current_tool = exec_tools[0] if exec_tools else "qa"

    return {
        "plan": plan,
        "current_tool": current_tool,
    }


def retrieve_node(state: AgentState):
    """🔍 检索节点：使用规划Agent改写后的查询进行检索"""
    plan = state.get("plan", {})
    # 优先使用规划Agent改写后的查询
    query = plan.get("rewritten_query", state["question"])

    print(f"\n🔍 [检索 Agent] 使用查询: '{query}'")

    context_chunks = api_retry_wrapper(retriever.search, query)

    return {"context": context_chunks}


def execute_node(state: AgentState):
    """
    ⚙️ 执行节点（改进！）
    - 根据规划结果，动态调用不同的工具方法
    - 不再是固定的 generate_answer，而是根据意图分发
    """
    question = state["question"]
    context_chunks = state["context"]
    feedback = state.get("feedback", "")
    chat_history = state.get("messages", [])
    current_tool = state.get("current_tool", "qa")

    print(f"\n⚙️ [执行 Agent] 调用工具: [{current_tool}] (反馈: {'有' if feedback else '无'})")

    # 核心：通过 dispatch_tool 动态分发到不同的生成策略
    answer = api_retry_wrapper(
        generator.dispatch_tool,
        current_tool,
        question=question,
        context_chunks=context_chunks,
        feedback=feedback,
        history=chat_history
    )

    return {"answer": answer}


def verify_node(state: AgentState):
    """🛡️ 验证节点：核查幻觉"""
    question = state["question"]
    context_chunks = state["context"]
    answer = state["answer"]
    current_count = state.get("retry_count", 0)

    print(f"\n🛡️ [验证 Agent] 审查中... (已重试: {current_count} 次)")

    feedback = api_retry_wrapper(generator.verify_answer, question, context_chunks, answer)

    if feedback == "PASS":
        print("   ✅ 审查通过！")
        return {"feedback": "PASS"}
    else:
        print(f"   ❌ 发现问题: {feedback}")
        return {"feedback": feedback, "retry_count": current_count + 1}


def error_node(state: AgentState):
    """🚨 兜底节点"""
    return {"answer": "抱歉，基于当前检索到的文献，系统经过多次尝试仍无法得出严谨的结论。"}


def should_continue(state: AgentState):
    """条件路由：决定流程走向"""
    feedback = state.get("feedback", "")
    retry_count = state.get("retry_count", 0)

    if feedback == "PASS":
        return END
    if retry_count >= 3:
        print("🚨 达到最大重试次数，触发熔断！")
        return "error"

    print(f"🔄 准备第 {retry_count + 1} 次反思重写...")
    return "execute"  # 注意：这里改名为 execute


# ==================== 构建 Agent 图 ====================

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("plan", plan_node)          # 🆕 新增规划节点
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("execute", execute_node)     # 重命名: analyze → execute
workflow.add_node("verify", verify_node)
workflow.add_node("error", error_node)

# 铺设边
workflow.add_edge(START, "plan")              # 🆕 入口先到规划节点
workflow.add_edge("plan", "retrieve")         # 规划完成后去检索
workflow.add_edge("retrieve", "execute")      # 检索后去执行
workflow.add_edge("execute", "verify")        # 执行后去验证

# 条件路由
workflow.add_conditional_edges(
    "verify",
    should_continue,
    {
        "execute": "execute",  # 打回重写
        "error": "error",      # 熔断
        END: END               # 通过
    }
)

workflow.add_edge("error", END)

# 编译图（带记忆）
app_graph = workflow.compile(checkpointer=memory)


# ==================== FastAPI 应用 ====================

app = FastAPI(title="PaperMind Agent API")

# 全局初始化
embedder = ZhipuEmbedder(api_key=ZHIPU_API_KEY)
retriever = PaperRetriever(embedder)
generator = ChatGenerator()
planner = PlannerAgent()  # 🆕 初始化规划Agent


def api_retry_wrapper(func, *args, **kwargs):
    """API 调用重试包装器（指数退避）"""
    max_retries = 3
    base_delay = 2
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ 限速重试 {attempt + 1}/{max_retries}，等待 {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise e


@app.get("/")
def read_root():
    return {"message": "PaperMind Agent API 启动成功！"}


@app.post("/chat")
def handle_chat(
    question: str = Form(...),
    thread_id: str = Form(...),
    file: UploadFile = File(None)
):
    # 处理文件上传
    if file is not None:
        temp_file_path = f"./temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(file.file.read())

        print(f"📦 论文已保存: {temp_file_path}")

        parser = PaperParser(temp_file_path)
        full_text = parser.extract_text()

        chunker = PaperChunker()
        chunks_dict = chunker.split_by_section(full_text)

        documents, metadatas, ids = [], [], []
        for section_name, chunk_list in chunks_dict.items():
            for chunk in chunk_list:
                documents.append(chunk)
                metadatas.append({"section": section_name})
                ids.append(str(uuid.uuid4()))

        # 重新构建索引
        global retriever, embedder
        embedder = ZhipuEmbedder(api_key=ZHIPU_API_KEY)
        retriever = PaperRetriever(embedder)
        retriever.build_index(documents, metadatas, ids)
        print(f"✅ 索引构建完成，共 {len(documents)} 个 Chunk")

    # 流式响应
    async def event_generator():
        initial_state = {"question": question}
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in app_graph.stream(initial_state, config, stream_mode="updates"):
            yield json.dumps(chunk, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
