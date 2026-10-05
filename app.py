import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Network & Sysadmin Assistant", page_icon="🖥️")
st.title("🖥️ Network & Sysadmin Assistant (RAG)")
st.caption("ผู้ช่วยตอบคำถามด้าน IT Infrastructure และการตั้งค่าอุปกรณ์เครือข่าย")

@st.cache_resource
def initialize_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

with st.spinner("กำลังโหลด Knowledge Base..."):
    try:
        vectorstore = initialize_rag()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}\nโปรดตรวจสอบว่ามีโฟลเดอร์ data/ และไฟล์ .txt อยู่ข้างใน")
        st.stop()

# '''llm = ChatGroq(
#     temperature=0, 
#     groq_api_key=st.secrets["GROQ_API_KEY"], 
#     model_name="gemma2-9b-it" 
# )'''

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    google_api_key=st.secrets["GEMINI_API_KEY"]
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""คุณคือ Senior Network Engineer จงตอบคำถามหรือให้คำแนะนำทางเทคนิคโดยอ้างอิงจากคู่มือ (Context) ด้านล่างนี้เท่านั้น
    
    กฎเหล็ก:
    1. หากข้อมูลใน Context ไม่มีคำตอบ ให้ตอบว่า: "ไม่พบข้อมูล" ห้ามเดาหรือสร้างคำสั่งขึ้นมาเองเด็ดขาด
    2. หากตอบได้ ให้แสดงขั้นตอนหรือ Command Line ออกมาให้ชัดเจน พร้อมระบุชื่อไฟล์อ้างอิงตอนท้าย
    
    Context:
    {context}
    
    คำถาม: {question}
    
    คำตอบ:"""
)

# เพิ่ม StrOutputParser() เข้าไปต่อท้าย เพื่อให้แปลงเป็นข้อความล้วน
chain = prompt_template | llm | StrOutputParser()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("พิมพ์คำถาม เช่น 'วิธีตั้งค่า VLAN 10' หรือ 'เงื่อนไขการต่อ Static Zone'"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        docs = retriever.invoke(user_input)
        context_text = "\n\n".join([f"[อ้างอิงจากไฟล์: {doc.metadata['source']}]\n{doc.page_content}" for doc in docs])
        
        with st.spinner("กำลังวิเคราะห์ข้อมูล..."):
            # ตอนนี้ chain จะคืนค่าเป็น text ธรรมดาโดยตรงเลย ไม่ต้องเรียก .content แล้ว
            answer = chain.invoke({"context": context_text, "question": user_input})
            st.markdown(answer)
            
            with st.expander("ดูเอกสารอ้างอิงที่ระบบใช้ (Retrieved Context)"):
                for i, doc in enumerate(docs):
                    st.write(f"**Chunk {i+1} (จาก {doc.metadata['source']})**")
                    st.info(doc.page_content)
            
        st.session_state.messages.append({"role": "assistant", "content": answer})