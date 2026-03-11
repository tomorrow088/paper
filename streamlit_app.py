"""
改进版 Streamlit 前端
新增：显示 Agent 的规划过程，让用户看到系统在"思考"
"""

import requests
import json
import time
import streamlit as st
import uuid

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📄 PaperMind 智能论文助手 (Agent版)")
st.caption("🤖 我能理解你的意图，自动选择最佳策略回答")

uploaded_file = st.sidebar.file_uploader("请上传你要阅读的论文", type=["pdf"])

# 历史记录渲染
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if len(st.session_state.messages) == 0:
    st.chat_message("assistant").write(
        "你好！我是你的论文阅读助手。你可以：\n"
        "- 📝 问我论文中的具体问题\n"
        "- 📊 让我总结某个章节\n"
        "- 🔍 让我对比不同方法\n"
        "- 📋 让我提取关键信息\n\n"
        "上传论文后开始吧！"
    )

# 意图图标映射
INTENT_ICONS = {
    "qa": "💬 问答模式",
    "summarize": "📝 总结模式",
    "compare": "📊 对比模式",
    "extract": "📋 提取模式",
}

prompt = st.chat_input("请输入问题...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        final_answer = ""

    if uploaded_file:
        files_data = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        form_data = {"question": prompt, "thread_id": st.session_state.thread_id}

        response = requests.post(
            "http://127.0.0.1:8000/chat",
            data=form_data,
            files=files_data,
            stream=True
        )
    else:
        form_data = {"question": prompt, "thread_id": st.session_state.thread_id}
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            data=form_data,
            stream=True
        )

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line.decode("utf-8"))
            node_name = list(chunk.keys())[0]
            data = chunk[node_name]

            # 🆕 处理规划节点的输出
            if node_name == "plan":
                plan = data.get("plan", {})
                intent = plan.get("intent", "qa")
                intent_label = INTENT_ICONS.get(intent, f"🔧 {intent}")
                reasoning = plan.get("reasoning", "")
                status_placeholder.info(
                    f"🧭 **规划完成** → {intent_label}\n\n"
                    f"策略: {reasoning}"
                )
                time.sleep(1)  # 让用户有时间看到规划结果

            elif node_name == "retrieve":
                status_placeholder.info("🔍 正在从论文库检索核心片段...")

            elif node_name == "execute":
                status_placeholder.info("⚙️ 正在基于文献生成回答...")
                if "answer" in data:
                    final_answer = data["answer"]

            elif node_name == "verify":
                if data.get("feedback") == "PASS":
                    status_placeholder.success("✅ 审查通过！")
                else:
                    status_placeholder.warning(
                        f"🛡️ 审查发现问题，正在重写... (反馈: {data.get('feedback', '')[:100]})"
                    )

            elif node_name == "error":
                final_answer = data.get("answer", "系统异常")
                status_placeholder.error("🚨 已达到最大重试次数。")

    # 清除状态，打字机效果展示答案
    status_placeholder.empty()
    answer_placeholder = st.empty()
    full_response = ""

    for char in final_answer:
        full_response += char
        answer_placeholder.markdown(full_response + "▌")
        time.sleep(0.03)

    answer_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
