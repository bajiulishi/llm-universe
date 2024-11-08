import streamlit as st
import os
import sys
sys.path.append("../C3 搭建知识库") # 将父目录放入系统路径中
from zhipuai_embedding import ZhipuAIEmbeddings
from zhipuai_llm import ZhipuAILLM

from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file

def get_vectordb():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = '../C3 搭建知识库/data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb

def generate_answer(question:str):
    # 1.构建本地数据库
    vectordb = get_vectordb()

    # 2.构建LLM
    api_key = "d2a91daa8a1004bb03209890cef9fb48.9tzT78TrSsBWjePU"
    zhipuai_llm = ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = api_key)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        zhipuai_llm,
        retriever = vectordb.as_retriever(),
        memory = memory
    )
    result = qa_chain({"question": question})
    return result['answer']

# Streamlit 应用程序界面
def main():
    st.title('动手学大模型应用开发')

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        answer = generate_answer(prompt)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   


if __name__ == "__main__":
    main()
    
