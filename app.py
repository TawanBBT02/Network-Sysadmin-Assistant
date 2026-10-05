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

st.set_page_config(
    page_title="Network & Sysadmin Assistant",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS : ธีมสว่าง ขาว + โทนร้อน
# ไอคอนทั้งหมดใช้ Material icons / SVG เส้นแบบเดียวกัน (ไม่ใช้อีโมจิ)
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --hot-orange: #FF6B35;
    --coral:      #FF8A5B;
    --amber:      #FFB347;
    --red-warm:   #F25C54;
    --cream:      #FFF6EE;
    --peach:      #FFE8D6;
    --ink:        #2D2A32;
    --muted:      #8A7F78;
}

html, body, .stApp, [data-testid="stMarkdownContainer"], button, textarea {
    font-family: 'Prompt', sans-serif;
}
.stApp { color: var(--ink); }

.stApp {
    background:
        radial-gradient(900px 400px at 100% -10%, rgba(255,179,71,0.16), transparent 60%),
        radial-gradient(700px 380px at -10% 0%, rgba(255,107,53,0.10), transparent 60%),
        #FFFFFF;
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { padding-top: 2.5rem; }

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1.4rem;
    border-radius: 22px;
    background: linear-gradient(120deg, #FF6B35 0%, #FF8A5B 55%, #FFB347 100%);
    color: #fff;
    box-shadow: 0 14px 34px rgba(255,107,53,0.26);
    animation: slideDown .7s cubic-bezier(.2,.8,.2,1) both;
}
.hero-icon {
    flex: 0 0 auto;
    width: 56px; height: 56px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 16px;
    background: rgba(255,255,255,0.22);
    border: 1px solid rgba(255,255,255,0.45);
    backdrop-filter: blur(4px);
    z-index: 1;
}
.hero-text { z-index: 1; }
.hero h1 {
    margin: 0; padding: 0;
    font-size: 1.55rem;
    font-weight: 600;
    line-height: 1.25;
    color: #fff;
}
.hero p {
    margin: .3rem 0 0 0;
    font-size: .92rem;
    font-weight: 300;
    color: rgba(255,255,255,.95);
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: .75rem;
    padding: .15rem .75rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.22);
    border: 1px solid rgba(255,255,255,0.5);
    font-size: .75rem;
    font-weight: 400;
}
.badge .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #fff;
    animation: pulse 1.6s infinite;
}
.hero::before, .hero::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    background: rgba(255,255,255,0.16);
    animation: floaty 7s ease-in-out infinite;
}
.hero::before { width: 150px; height: 150px; right: -30px; top: -55px; }
.hero::after  { width: 90px;  height: 90px;  right: 110px; bottom: -45px; animation-delay: 1.5s; }

/* ---------- Chat bubbles ---------- */
[data-testid="stChatMessage"] {
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
    border-radius: 18px;
    border: 1px solid var(--peach);
    background: #FFFFFF;
    box-shadow: 0 4px 16px rgba(255,107,53,0.07);
    animation: popIn .45s ease both;
    transition: transform .2s ease, box-shadow .2s ease;
}
[data-testid="stChatMessage"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(255,107,53,0.14);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #FFF8F2, #FFEBDC);
    border-color: #FFD6BA;
}

/* Avatar เป็นกล่องไล่สี + ไอคอน Material สีขาว (ตรงกลางพอดี) */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    width: 2.3rem; height: 2.3rem;
    display: flex; align-items: center; justify-content: center;
    border-radius: 12px;
    border: none;
    color: #fff;
}
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #FFB347, #FF8A5B);
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #FF6B35, #F25C54);
}
[data-testid="stChatMessageAvatarUser"] *,
[data-testid="stChatMessageAvatarAssistant"] * {
    color: #fff !important;
    font-size: 1.3rem;
}

/* โค้ด / command */
code, pre, [data-testid="stCodeBlock"] { font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stCodeBlock"], pre {
    border-left: 4px solid var(--hot-orange);
    border-radius: 10px;
    background: #FFF9F4 !important;
}
:not(pre) > code {
    color: var(--red-warm);
    background: #FFF0E6;
    padding: 2px 6px;
    border-radius: 6px;
}

/* ---------- Chat input ---------- */
[data-testid="stChatInput"] {
    border-radius: 18px;
    border: 2px solid var(--peach);
    background: #fff;
    box-shadow: 0 6px 20px rgba(255,107,53,0.10);
    transition: border-color .25s, box-shadow .25s;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea { background: #fff !important; }
[data-testid="stChatInput"]:focus-within {
    border-color: var(--hot-orange);
    box-shadow: 0 0 0 4px rgba(255,107,53,0.16);
}
[data-testid="stChatInputSubmitButton"] { color: var(--hot-orange); }

/* ---------- Buttons (กว้างเท่ากัน + ไอคอนตรงแนว) ---------- */
.stButton, [data-testid="stButton"] { width: 100%; }
.stButton > button, [data-testid="stButton"] > button {
    width: 100%;
    min-height: 2.6rem;
    padding: .5rem .9rem;
    border-radius: 14px;
    border: 1px solid var(--peach);
    background: #fff;
    color: var(--ink);
    font-size: .88rem;
    font-weight: 400;
    transition: all .22s ease;
}
.stButton > button > div, [data-testid="stButton"] > button > div {
    width: 100%;
    justify-content: flex-start;
    gap: .55rem;
    text-align: left;
}
.stButton > button [data-testid="stIconMaterial"],
[data-testid="stButton"] > button [data-testid="stIconMaterial"] {
    color: var(--hot-orange);
    font-size: 1.25rem;
    transition: color .22s ease;
}
.stButton > button:hover, [data-testid="stButton"] > button:hover {
    border-color: var(--hot-orange);
    color: #fff;
    background: linear-gradient(120deg, var(--hot-orange), var(--amber));
    transform: translateX(4px);
    box-shadow: 0 8px 18px rgba(255,107,53,0.24);
}
.stButton > button:hover [data-testid="stIconMaterial"],
[data-testid="stButton"] > button:hover [data-testid="stIconMaterial"] { color: #fff; }
.stButton > button:active { transform: scale(.97); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFF6EE 0%, #FFFFFF 100%);
    border-right: 1px solid var(--peach);
}
[data-testid="stSidebar"] hr { border-color: var(--peach); }
.side-title {
    display: flex; align-items: center; gap: .45rem;
    font-weight: 600; font-size: 1rem;
    color: var(--hot-orange);
    margin: .2rem 0 .8rem 0;
}
.side-note { font-size: .8rem; color: var(--muted); line-height: 1.55; }

/* ---------- Expander / Alert ---------- */
[data-testid="stExpander"] {
    border: 1px dashed #FFC9A3;
    border-radius: 14px;
    background: #FFFCF9;
}
[data-testid="stExpander"] summary:hover { color: var(--hot-orange); }
[data-testid="stAlert"] {
    background: #FFF4EA;
    color: var(--ink);
    border: 1px solid #FFD9BD;
    border-radius: 12px;
}
.stSpinner > div > div { border-top-color: var(--hot-orange) !important; }
::-webkit-scrollbar { width: 9px; }
::-webkit-scrollbar-thumb { background: #FFC9A3; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--hot-orange); }

/* ---------- Empty state ---------- */
.empty { text-align: center; padding: 1.6rem 1rem 1rem; color: var(--muted); animation: popIn .6s ease both; }
.empty-icon {
    width: 72px; height: 72px;
    margin: 0 auto .9rem;
    display: flex; align-items: center; justify-content: center;
    border-radius: 22px;
    background: linear-gradient(135deg, #FFF0E4, #FFE0C8);
    border: 1px solid #FFD2B0;
    box-shadow: 0 10px 24px rgba(255,107,53,0.16);
    animation: bounce 2.6s ease-in-out infinite;
}
.empty p { margin: 0; line-height: 1.7; font-size: .95rem; }

/* ---------- Keyframes ---------- */
@keyframes slideDown { from {opacity:0; transform: translateY(-18px);} to {opacity:1; transform:none;} }
@keyframes popIn     { from {opacity:0; transform: translateY(10px) scale(.98);} to {opacity:1; transform:none;} }
@keyframes floaty    { 0%,100% {transform: translateY(0);} 50% {transform: translateY(14px);} }
@keyframes pulse     { 0% {box-shadow:0 0 0 0 rgba(255,255,255,.8);} 70% {box-shadow:0 0 0 8px rgba(255,255,255,0);} 100% {box-shadow:0 0 0 0 rgba(255,255,255,0);} }
@keyframes bounce    { 0%,100% {transform: translateY(0);} 50% {transform: translateY(-8px);} }
</style>
""",
    unsafe_allow_html=True,
)

# SVG ไอคอนเส้น (ชุดเดียวกันทั้งหมด)
ICON_NETWORK = (
    '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="{c}" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="9" y="2" width="6" height="6" rx="1.2"/><rect x="2" y="16" width="6" height="6" rx="1.2"/>'
    '<rect x="16" y="16" width="6" height="6" rx="1.2"/>'
    '<path d="M12 8v4M5 16v-2a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v2"/></svg>'
)
ICON_BOLT = (
    '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#FF6B35" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>'
)

# ------------------------------------------------------------------
# Hero header
# ------------------------------------------------------------------
st.markdown(
    f"""<div class="hero">
<div class="hero-icon">{ICON_NETWORK.format(s=30, c="#fff")}</div>
<div class="hero-text">
<h1>Network &amp; Sysadmin Assistant</h1>
<p>ผู้ช่วยตอบคำถามด้าน IT Infrastructure และการตั้งค่าอุปกรณ์เครือข่าย</p>
<span class="badge"><span class="dot"></span>RAG · ตอบจากคู่มือในระบบเท่านั้น</span>
</div>
</div>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# RAG (ตรรกะเดิม)
# ------------------------------------------------------------------
@st.cache_resource
def initialize_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

with st.spinner("กำลังโหลด Knowledge Base..."):
    try:
        vectorstore = initialize_rag()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}\nโปรดตรวจสอบว่ามีโฟลเดอร์ data/ และไฟล์ .txt อยู่ข้างใน")
        st.stop()

# '''llm = ChatGroq(
#     temperature=0, 
#     groq_api_key=st.secrets["GROQ_API_KEY"], 
#     model_name="gemma2-9b-it" 
# )'''

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
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

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "awaiting_answer" not in st.session_state:
    st.session_state.awaiting_answer = False

USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/smart_toy:"

# ------------------------------------------------------------------
# Sidebar : คำถามตัวอย่าง + ปุ่มล้างแชท (ไอคอน Material)
# ------------------------------------------------------------------
EXAMPLES = [
    (":material/lan:",            "วิธีตั้งค่า VLAN 10 บน Cisco"),
    (":material/alt_route:",      "ตั้งค่า OSPF เบื้องต้นทำยังไง"),
    (":material/shield:",         "เงื่อนไขการต่อ Static Zone"),
    (":material/restart_alt:",    "วิธีทำ Factory Reset สวิตช์ Aruba"),
    (":material/dns:",            "ตั้งค่า HA ด้วย Keepalived"),
    (":material/network_check:",  "ใช้ ping / traceroute / arp แก้ปัญหา"),
]

with st.sidebar:
    st.markdown(f'<div class="side-title">{ICON_BOLT}<span>ลองถามดู</span></div>', unsafe_allow_html=True)
    for i, (icon, text) in enumerate(EXAMPLES):
        if st.button(text, key=f"ex_{i}", icon=icon):
            st.session_state.pending_question = text

    st.markdown("---")
    if st.button("ล้างประวัติการสนทนา", key="clear", icon=":material/delete_sweep:"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.awaiting_answer = False
        st.rerun()

    st.markdown(
        '<p class="side-note">ระบบจะตอบจากคู่มือในโฟลเดอร์ <code>data/</code> เท่านั้น '
        'หากไม่มีข้อมูลจะตอบว่า "ไม่พบข้อมูล"</p>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# รับคำถามก่อนวาดหน้า (แก้ปัญหา Empty state ค้างตอนมีแชทแล้ว)
# ------------------------------------------------------------------
typed_input = st.chat_input("พิมพ์คำถาม เช่น 'วิธีตั้งค่า VLAN 10' หรือ 'เงื่อนไขการต่อ Static Zone'")
user_input = typed_input or st.session_state.pending_question
st.session_state.pending_question = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.awaiting_answer = True

# ------------------------------------------------------------------
# แสดง Empty state หรือประวัติแชท
# ------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        f"""<div class="empty">
<div class="empty-icon">{ICON_NETWORK.format(s=34, c="#FF6B35")}</div>
<p>สวัสดีครับ พิมพ์คำถามด้านเครือข่ายหรือเซิร์ฟเวอร์ได้เลย<br>หรือกดเลือกคำถามตัวอย่างจากแถบด้านซ้าย</p>
</div>""",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# สร้างคำตอบ (หลังแสดงคำถามของผู้ใช้แล้ว)
# ------------------------------------------------------------------
if st.session_state.awaiting_answer:
    question = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        docs = retriever.invoke(question)
        context_text = "\n\n".join([f"[อ้างอิงจากไฟล์: {doc.metadata['source']}]\n{doc.page_content}" for doc in docs])

        with st.spinner("กำลังวิเคราะห์ข้อมูล..."):
            # chain คืนค่าเป็น text ธรรมดาโดยตรง ไม่ต้องเรียก .content
            answer = chain.invoke({"context": context_text, "question": question})
            st.markdown(answer)

            with st.expander("ดูเอกสารอ้างอิงที่ระบบใช้ (Retrieved Context)"):
                for i, doc in enumerate(docs):
                    st.write(f"**Chunk {i+1}** · `{os.path.basename(doc.metadata['source'])}`")
                    st.info(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.awaiting_answer = False