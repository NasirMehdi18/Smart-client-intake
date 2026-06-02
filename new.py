import streamlit as st
import os, uuid, sqlite3, json
from typing import Optional
from dotenv import load_dotenv

# ================== ENV ==================
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("Hugging_Face_api", "")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

# ================== CONSTANTS ==================
KNOWLEDGE_BASE_PATH = r"D:\New folder\Laptop Data\lawAssistant\knowledge_base\vector_db"
DB_PATH = "chats.db"

SYSTEM_PROMPT = """
آپ ایک پاکستانی لیگل اسسٹنٹ ہیں۔ جواب ہمیشہ خالص اور درست اردو میں دیں۔

جواب کا فارمیٹ:
1) قانونی وضاحت
2) متعلقہ قوانین / دفعات / آرٹیکلز
3) قانونی وجوہات
4) عملی مشورہ

Context دستیاب ہو تو لازماً استعمال کریں۔
"""

SUMMARY_PROMPT = """
صارف کے اس سوال سے 2-4 الفاظ میں ایک مختصر، واضح اور یاد رکھنے والی اردو سمری بنائیں جو چیٹ کا نام بن سکے۔
صرف سمری دیں، کچھ اور نہ لکھیں۔ مثال:
سوال: پاکستان میں طلاق کے بعد بچے کی حضانت کا قانون کیا ہے؟
سمری: طلاق اور حضانت

سوال: {question}
سمری:
"""

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Smart Client InTake and Legal Issue Identifier", page_icon="⚖️", layout="wide", initial_sidebar_state="collapsed")

# ================== THEME ==================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def load_css():
    dark = st.session_state.theme == "dark"
    bg = "radial-gradient(circle at top,#0f172a,#020617)" if dark else "linear-gradient(to bottom,#f8fafc,#e5e7eb)"
    text = "#e5e7eb" if dark else "#020617"
    panel = "rgba(30,41,59,.9)" if dark else "#ffffff"
    user_bg = "linear-gradient(135deg,#2563eb,#1e40af)" if dark else "linear-gradient(135deg,#6366f1,#4f46e5)"
    border = "#334155" if dark else "#cbd5e1"

    st.markdown(f"""
    <style>
    .stApp {{ background:{bg}; color:{text}; font-family:'Segoe UI','Noto Nastaliq Urdu'; }}
    .chat-row {{ display:flex; margin-bottom:16px }}
    .chat-user {{ margin-left:auto; background:{user_bg}; color:white; padding:14px 18px;
        border-radius:18px 18px 4px 18px; max-width:70%; line-height:1.9; box-shadow:0 8px 22px rgba(37,99,235,.35); }}
    .chat-ai {{ margin-right:auto; background:{panel}; color:{text}; padding:16px 20px;
        border-radius:18px 18px 18px 4px; max-width:75%; line-height:2; box-shadow:0 8px 25px rgba(0,0,0,.55); }}
    .chat-label {{ font-size:13px; opacity:.65; margin-bottom:6px }}
    .stTextInput > div > div > input {{ background:#020617; color:#e5e7eb; border:1px solid {border}; border-radius:14px; padding:12px; }}
    .chat-container {{ max-height: 70vh; overflow-y: auto; padding-right: 10px; }}
    </style>
    """, unsafe_allow_html=True)

load_css()

# ================== VECTOR STORE ==================
@st.cache_resource
def load_vectorstore() -> Optional[FAISS]:
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        return FAISS.load_local(KNOWLEDGE_BASE_PATH, emb, allow_dangerous_deserialization=True)
    return None

# ================== LLM ==================
def get_llm(repo_id, temperature, max_tokens):
    base = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=temperature,
        max_new_tokens=max_tokens,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        timeout=300,
        # provider="cerebras"
        provider="novita"
    )
    return ChatHuggingFace(llm=base)

# ================== DATABASE ==================
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS sessions(
        session_id TEXT PRIMARY KEY,
        name TEXT,
        messages TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    con.commit(); con.close()

def load_sessions():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT session_id, name, messages FROM sessions ORDER BY created_at DESC")
    rows = cur.fetchall()
    con.close()
    return {sid: {"name": name, "messages": json.loads(msgs)} for sid, name, msgs in rows}

def save_session(sid, name, messages):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("REPLACE INTO sessions (session_id, name, messages) VALUES (?,?,?)", (sid, name, json.dumps(messages)))
    con.commit(); con.close()

def delete_session(sid):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
    con.commit(); con.close()

init_db()

# ================== STATE ==================
if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()
if "active_session" not in st.session_state:
    st.session_state.active_session = None
if "input_key" not in st.session_state:
    st.session_state.input_key = str(uuid.uuid4())
if "chat_search" not in st.session_state:
    st.session_state.chat_search = ""

# ================== HELPERS ==================
def new_chat():
    sid = str(uuid.uuid4())
    st.session_state.sessions[sid] = {"name": "نئی چیٹ", "messages": [], "saved": False}
    st.session_state.active_session = sid

if st.session_state.active_session is None:
    new_chat()

# ================== TOP BAR (Theme Toggle) ==================
col1, col2 = st.columns([6,1])
with col2:
    if st.button("🌙" if st.session_state.theme == "dark" else "☀️", key="theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

# ================== SIDEBAR ==================
with st.sidebar:
    st.title("⚖️ LawGPT")

    if st.button("➕ نئی چیٹ"):
        new_chat(); st.rerun()

    st.text_input("🔍 چیٹ تلاش کریں", key="chat_search", value=st.session_state.chat_search)

    st.markdown("---")

    # Latest chats first
    filtered_items = [(sid, s) for sid, s in st.session_state.sessions.items()
                      if st.session_state.chat_search.lower() in s['name'].lower()]

    for sid, sess in filtered_items:
        c1, c2 = st.columns([6,1])
        with c1:
            if st.button(sess['name'], key=f"select_{sid}"):
                st.session_state.active_session = sid
                st.rerun()
        with c2:
            with st.popover("⋮"):
                new_name = st.text_input("نام تبدیل کریں", value=sess['name'], key=f"rename_{sid}")
                if st.button("✏️ تبدیل کریں", key=f"rename_btn_{sid}"):
                    st.session_state.sessions[sid]['name'] = new_name
                    save_session(sid, new_name, sess['messages'])
                    st.rerun()
                st.markdown("---")
                if st.button("🗑 ڈیلیٹ", key=f"delete_{sid}"):
                    delete_session(sid)
                    del st.session_state.sessions[sid]
                    if st.session_state.active_session == sid:
                        new_chat()
                    st.rerun()

    st.markdown("---")
    model_id = st.text_input("Model ID", "meta-llama/Llama-3.1-8B-Instruct")
    temp = st.slider("Temperature", 0.0, 1.0, 0.3)
    max_tokens = st.slider("Max Tokens", 500, 4000, 1500, 100)

# ================== MAIN ==================
st.markdown("## ⚖️ Smart Client InTake and Legal Issue Identifier")
st.caption("خالص اردو قانونی رہنمائی – پاکستان کے قوانین کے مطابق")

# Chat container with auto-scroll
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

msgs = st.session_state.sessions[st.session_state.active_session]['messages']
for m in msgs:
    if m['role'] == 'user':
        st.markdown(f"<div class='chat-row'><div class='chat-user'><div class='chat-label'>🧑 آپ</div>{m['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-row'><div class='chat-ai'><div class='chat-label'>⚖️ Smart Client Intake</div>{m['content']}</div></div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Auto-scroll to bottom
st.markdown("""
<script>
    const container = parent.document.querySelector('.chat-container');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
</script>
""", unsafe_allow_html=True)

q = st.text_input("اپنا قانونی سوال لکھیں", key=st.session_state.input_key, placeholder="مثال: پاکستان میں طلاق کے بعد بچے کی حضانت کا قانون کیا ہے؟")

if st.button("🚀 جواب حاصل کریں") and q.strip():
    with st.spinner("قانونی جواب تیار ہو رہا ہے..."):
        # پہلا سوال ہے تو سمری جنریٹ کرو
        if len(msgs) == 0:
            summary_llm = get_llm(model_id, 0.1, 50)
            summary_res = summary_llm.invoke([HumanMessage(content=SUMMARY_PROMPT.format(question=q))])
            auto_name = summary_res.content.strip()
            if not auto_name or len(auto_name) > 30:
                auto_name = "قانونی سوال"
            st.session_state.sessions[st.session_state.active_session]['name'] = auto_name

        msgs.append({"role": "user", "content": q})

        context = ""
        vs = load_vectorstore()
        if vs:
            docs = vs.similarity_search(q, k=4)
            context = "\n".join(d.page_content for d in docs)

        llm = get_llm(model_id, temp, max_tokens)
        res = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"سوال: {q}\n\nContext:\n{context}")
        ])
        msgs.append({"role": "assistant", "content": res.content})

        sid = st.session_state.active_session
        name = st.session_state.sessions[sid]['name']
        st.session_state.sessions[sid]['saved'] = True
        save_session(sid, name, msgs)

    st.session_state.input_key = str(uuid.uuid4())
    st.rerun()