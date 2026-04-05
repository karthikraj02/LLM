import streamlit as st
import requests

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Custom LLM Chat",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Claude-like light beige interface ────────────────────────────
st.markdown("""
<style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #FAF9F6 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #1a1a1a !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #F3F2EE !important;
    border-right: 1px solid #e0ddd8 !important;
    min-width: 220px !important;
    max-width: 220px !important;
}
[data-testid="stSidebar"] * { color: #1a1a1a !important; }
[data-testid="stSidebarContent"] { padding: 1rem 0.75rem !important; }

/* ── Sidebar navigation header ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.65rem 1rem;
    font-weight: 600;
    font-size: 0.95rem;
    color: #1a1a1a;
}
.sidebar-logo-icon {
    font-size: 1.2rem;
    color: #c0392b;
}
.sidebar-nav-btn {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.65rem;
    border-radius: 8px;
    font-size: 0.875rem;
    color: #6b6b6b;
    cursor: pointer;
    transition: background 0.15s;
    margin-bottom: 0.2rem;
}
.sidebar-nav-btn:hover { background: #ece9e4; }
.sidebar-nav-btn.active {
    background: #e0ddd8;
    color: #1a1a1a;
    font-weight: 500;
}

/* ── Slider labels ── */
[data-testid="stSidebar"] label {
    font-size: 0.8rem !important;
    color: #6b6b6b !important;
}
[data-testid="stSidebar"] .stSlider { padding: 0 !important; }
[data-testid="stSidebar"] h3 {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #9a9a9a !important;
    margin-bottom: 0.75rem !important;
    font-weight: 600 !important;
}

/* ── Top header bar ── */
.top-header {
    position: fixed;
    top: 0;
    left: 220px;
    right: 0;
    height: 52px;
    background: #FAF9F6;
    border-bottom: 1px solid #e0ddd8;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding: 0 1.5rem;
    z-index: 999;
}
.plan-badge {
    font-size: 0.78rem;
    color: #6b6b6b;
    background: #F3F2EE;
    border: 1px solid #e0ddd8;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
}
.plan-badge a {
    color: #c0392b;
    text-decoration: none;
    font-weight: 500;
}

/* ── Main content area ── */
.main-area {
    margin-top: 52px;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: calc(100vh - 52px);
    padding: 0 1rem;
}

/* ── Greeting ── */
.greeting-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 0 2rem;
    gap: 0.5rem;
}
.greeting-spark {
    font-size: 2rem;
    color: #c0392b;
    line-height: 1;
}
.greeting-title {
    font-size: 1.75rem;
    font-weight: 500;
    color: #1a1a1a;
    letter-spacing: -0.02em;
    margin: 0;
}

/* ── Chat messages ── */
.chat-messages {
    width: 100%;
    max-width: 680px;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    margin-bottom: 1rem;
}
.chat-msg {
    display: flex;
    gap: 0.75rem;
}
.chat-msg.user { flex-direction: row-reverse; }
.msg-avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
    flex-shrink: 0;
    background: #e0ddd8;
    color: #6b6b6b;
}
.chat-msg.user .msg-avatar {
    background: #1a1a1a;
    color: #fff;
}
.msg-content {
    background: #f0ede8;
    padding: 0.7rem 1rem;
    border-radius: 12px;
    font-size: 0.925rem;
    line-height: 1.6;
    color: #1a1a1a;
    max-width: 85%;
    border-bottom-left-radius: 4px;
}
.chat-msg.user .msg-content {
    background: #1a1a1a;
    color: #fff;
    border-bottom-left-radius: 12px;
    border-bottom-right-radius: 4px;
}

/* ── Input box ── */
.input-section {
    width: 100%;
    max-width: 680px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    padding-bottom: 1.5rem;
}
.input-box-wrapper {
    width: 100%;
    background: #fff;
    border: 1.5px solid #e0ddd8;
    border-radius: 16px;
    padding: 0.65rem 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}
.input-box-wrapper:focus-within {
    border-color: #c0bdb8;
    box-shadow: 0 2px 12px rgba(0,0,0,0.12);
}
.model-tag {
    font-size: 0.75rem;
    color: #6b6b6b;
    background: #F3F2EE;
    border: 1px solid #e0ddd8;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    white-space: nowrap;
    font-weight: 500;
}

/* ── Tools banner ── */
.tools-banner {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: #9a9a9a;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    border: 1px solid #e0ddd8;
    background: #FAF9F6;
    cursor: pointer;
    user-select: none;
}

/* ── Streamlit widget overrides ── */
[data-testid="stTextInput"] > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
[data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important;
    color: #1a1a1a !important;
    font-size: 0.95rem !important;
    font-family: inherit !important;
    padding: 0 !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #9a9a9a !important; }
[data-testid="stTextInput"] input:focus { outline: none !important; box-shadow: none !important; }

/* Remove Streamlit button default styling */
.stButton button {
    background: #1a1a1a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50% !important;
    width: 34px !important;
    height: 34px !important;
    padding: 0 !important;
    min-width: unset !important;
    font-size: 1rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.15s !important;
}
.stButton button:hover { background: #333 !important; }
.stButton button:disabled { background: #e0ddd8 !important; color: #9a9a9a !important; }

/* Column gaps */
[data-testid="stHorizontalBlock"] { gap: 0 !important; align-items: center !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="sidebar-logo-icon">✦</span>
        <span>Custom LLM</span>
    </div>
    <div class="sidebar-nav-btn active">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        New Chat
    </div>
    <div class="sidebar-nav-btn">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polyline points="1 4 1 10 7 10"/>
            <path d="M3.51 15a9 9 0 1 0 .49-4.5"/>
        </svg>
        History
    </div>
    <hr style="border:none;border-top:1px solid #e0ddd8;margin:0.75rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("### Parameters")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
    max_tokens = st.slider("Max Tokens", 10, 500, 100, 10)
    top_k = st.slider("Top K", 1, 100, 40, 1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)

# ── Top header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <span class="plan-badge">Free plan &middot; <a href="#">Upgrade</a></span>
</div>
""", unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-area">', unsafe_allow_html=True)

# Greeting (shown when no messages)
if not st.session_state.messages:
    st.markdown("""
    <div class="greeting-section">
        <span class="greeting-spark">&#10022;</span>
        <h1 class="greeting-title">Back at it, karthik</h1>
    </div>
    """, unsafe_allow_html=True)

# Chat messages
if st.session_state.messages:
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role_class = "user" if msg["role"] == "user" else "bot"
        avatar_char = "K" if msg["role"] == "user" else "✦"
        content_html = msg["content"].replace("\n", "<br>")
        st.markdown(f"""
        <div class="chat-msg {role_class}">
            <div class="msg-avatar">{avatar_char}</div>
            <div class="msg-content">{content_html}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Input box ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="input-section">
    <div class="input-box-wrapper">
""", unsafe_allow_html=True)

col_attach, col_input, col_model, col_send = st.columns([0.05, 0.75, 0.12, 0.08])

with col_attach:
    st.markdown("""
    <div style="width:30px;height:30px;border-radius:50%;border:1.5px solid #e0ddd8;
         display:flex;align-items:center;justify-content:center;cursor:pointer;color:#6b6b6b;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
             stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)

with col_input:
    user_input = st.text_input(
        label="message",
        label_visibility="collapsed",
        placeholder="Message Custom LLM...",
        key="chat_input",
    )

with col_model:
    st.markdown('<span class="model-tag">Sonnet 4.6</span>', unsafe_allow_html=True)

with col_send:
    send_clicked = st.button("↑", key="send_btn")

st.markdown("""
    </div>
    <div class="tools-banner">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94
                     l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
        </svg>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
        </svg>
        <span class="tools-banner-text">Connect your tools to Custom LLM</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Handle send ───────────────────────────────────────────────────────────────
if (send_clicked or user_input) and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        resp = requests.post(
            "http://localhost:8000/generate",
            json={
                "prompt": user_input,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_k": top_k,
                "top_p": top_p,
            },
            timeout=60,
        )
        resp.raise_for_status()
        answer = resp.json().get("generated_text", "")
    except Exception as e:
        answer = f"Error connecting to the model server: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
