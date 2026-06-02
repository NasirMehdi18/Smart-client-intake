import streamlit as st
import os, uuid, sqlite3, json
from typing import Optional
from dotenv import load_dotenv

# ================== ENV & IMPORTS ==================
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

# ================== DATABASE FUNCTIONS (created_at updates on save) ==================
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS sessions(
        session_id TEXT PRIMARY KEY,
        name TEXT,
        messages TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    # پرانی DB کے لیے created_at ایڈ کرو اگر نہ ہو
    cur.execute("PRAGMA table_info(sessions)")
    columns = [info[1] for info in cur.fetchall()]
    if 'created_at' not in columns:
        cur.execute("ALTER TABLE sessions ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    con.commit()
    con.close()

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
    cur.execute("""INSERT INTO sessions (session_id, name, messages) 
                   VALUES (?,?,?) 
                   ON CONFLICT(session_id) DO UPDATE SET 
                   name=excluded.name, 
                   messages=excluded.messages,
                   created_at=CURRENT_TIMESTAMP""",
                (sid, name, json.dumps(messages)))
    con.commit()
    con.close()

def delete_session(sid):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
    con.commit()
    con.close()

init_db()

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Smart Client InTake and Legal Issue Identifier", page_icon="⚖️", layout="wide", initial_sidebar_state="collapsed")

# ================== THEME ==================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

dark = st.session_state.theme == "dark"

# ================== PROFESSIONAL CSS (Fixed Bottom Input + Auto-scroll) ==================
st.markdown(f"""
<style>
    html, body, .stApp {{
        height: 100%;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        background: {'#0f172a' if dark else '#f8fafc'};
        color: {'#e2e8f0' if dark else '#1e293b'};
        font-family: 'Segoe UI', 'Noto Nastaliq Urdu', sans-serif;
    }}
    .header-bar {{
        background: {'#1e293b' if dark else '#ffffff'};
        padding: 1rem 2rem;
        border-bottom: 1px solid {'#334155' if dark else '#e2e8f0'};
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}
    .main-container {{
        flex: 1;
        display: flex;
        flex-direction: column;
        max-width: 900px;
        margin: 0 auto;
        width: 100%;
        overflow: hidden;
    }}
    .chat-container {{
        flex: 1;
        overflow-y: auto;
        padding: 2rem;
        padding-bottom: 120px; /* input area کی جگہ */
    }}
    .message {{ margin-bottom: 1.5rem; display: flex; flex-direction: column; }}
    .user-message {{ align-items: flex-end; }}
    .ai-message {{ align-items: flex-start; }}
    .bubble {{
        max-width: 70%;
        padding: 1rem 1.5rem;
        border-radius: 20px;
        line-height: 1.8;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    .user-bubble {{
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border-bottom-right-radius: 6px;
    }}
    .ai-bubble {{
        background: {'#1e293b' if dark else '#f1f5f9'};
        color: {'#e2e8f0' if dark else '#1e293b'};
        border-bottom-left-radius: 6px;
    }}
    .input-area {{
        background: {'#0f172a' if dark else '#ffffff'};
        padding: 1rem 2rem;
        border-top: 1px solid {'#334155' if dark else '#e2e8f0'};
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        max-width: 900px;
        margin: 0 auto;
        z-index: 1000;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.3);
    }}
    .stTextInput > div > div > input {{
        background: {'#1e293b' if dark else '#f1f5f9'} !important;
        color: {'#e2e8f0' if dark else '#1e293b'} !important;
        border-radius: 20px !important;
        padding: 14px 20px !important;
        border: none !important;
    }}
</style>
""", unsafe_allow_html=True)

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
        provider="cerebras"
    )
    return ChatHuggingFace(llm=base)

# ================== STATE & HELPERS ==================
if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()
if "active_session" not in st.session_state:
    st.session_state.active_session = None
if "input_key" not in st.session_state:
    st.session_state.input_key = str(uuid.uuid4())
if "chat_search" not in st.session_state:
    st.session_state.chat_search = ""

def new_chat():
    sid = str(uuid.uuid4())
    st.session_state.sessions[sid] = {"name": "نئی چیٹ", "messages": []}
    st.session_state.active_session = sid

if st.session_state.active_session is None:
    new_chat()

# ================== HEADER BAR ==================
st.markdown('<div class="header-bar">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([4,3,1])
with col1:
    st.markdown("### ⚖️ Smart Client InTake and Legal Issue Identifier")
with col2:
    st.caption("خالص اردو قانونی رہنمائی – پاکستان کے قوانین کے مطابق")
with col3:
    if st.button("🌙" if dark else "☀️", key="theme_toggle"):
        st.session_state.theme = "light" if dark else "dark"
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ================== MAIN CONTAINER ==================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ================== SIDEBAR ==================
with st.sidebar:
    st.title("⚖️ LawGPT")

    if st.button("➕ نئی چیٹ"):
        new_chat()
        st.rerun()

    st.text_input("🔍 چیٹ تلاش کریں", key="chat_search", value=st.session_state.chat_search)

    st.markdown("---")

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

# ================== CHAT DISPLAY ==================
st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)

msgs = st.session_state.sessions[st.session_state.active_session]['messages']
for m in msgs:
    if m['role'] == 'user':
        st.markdown(f"""
        <div class="message user-message">
            <div class="bubble user-bubble">
                {m['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message ai-message">
            <div class="bubble ai-bubble">
                {m['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Auto-scroll to bottom
st.markdown("""
<script>
    const container = parent.document.querySelector('#chat-container');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
</script>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # main-container close

# ================== FIXED BOTTOM INPUT AREA ==================
st.markdown('<div class="input-area">', unsafe_allow_html=True)
col1, col2 = st.columns([6,1])
with col1:
    q = st.text_input("", key=st.session_state.input_key, placeholder="اپنا قانونی سوال یہاں لکھیں...", label_visibility="collapsed")
with col2:
    send = st.button("🚀")

if send and q.strip():
    with st.spinner("قانونی جواب تیار ہو رہا ہے..."):
        # پہلا سوال → آٹو سمری سے چیٹ نام
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
        save_session(sid, name, msgs)

    st.session_state.input_key = str(uuid.uuid4())
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)