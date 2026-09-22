"""
DR. MIGI — Clinical Intelligence & AI Healthcare Companion Web Interface
========================================================================
A clean, conversational Streamlit UI for DR. MIGI supporting both direct
clinical LLM chat and patient-grounded RAG analysis.
"""

import os
import sys
import json
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference.engine import DrMigiEngine
from backend.rag.pipeline import DrMigiRAGPipeline

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & High-Contrast Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DR. MIGI — Clinical Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS Theme (Fixes all text visibility issues)
st.markdown("""
<style>
    /* Main App Background & Base Typography */
    .stApp {
        background-color: #0A0E17 !important;
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F1626 !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }

    /* Global Text & Markdown Visibility Fixes */
    h1, h2, h3, h4, h5, h6, p, span, label, li {
        color: #F8FAFC !important;
    }

    /* Header Container */
    .migi-header {
        text-align: center;
        padding: 20px 10px 10px 10px;
        margin-bottom: 20px;
        border-bottom: 1px solid #1E293B;
    }
    .migi-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .migi-subtitle {
        font-size: 1.05rem;
        color: #94A3B8 !important;
        font-weight: 400;
    }

    /* Status Badge */
    .status-badge-container {
        display: flex;
        justify-content: center;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-general {
        background-color: rgba(56, 189, 248, 0.12);
        border: 1px solid #0284C7;
        color: #38BDF8 !important;
    }
    .badge-patient {
        background-color: rgba(129, 140, 248, 0.15);
        border: 1px solid #6366F1;
        color: #A5B4FC !important;
    }

    /* Chat Message Bubbles */
    [data-testid="stChatMessage"] {
        background-color: #111827 !important;
        border: 1px solid #1E293B !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }

    /* Assistant Chat Bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #0E1726 !important;
        border-left: 4px solid #38BDF8 !important;
    }

    /* User Chat Bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #182235 !important;
        border-left: 4px solid #818CF8 !important;
    }

    /* Chat Message Markdown Content */
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
        color: #F8FAFC !important;
        font-size: 0.98rem !important;
        line-height: 1.6 !important;
    }

    /* Chat Input Styling */
    [data-testid="stChatInput"] {
        background-color: #0A0E17 !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #161F30 !important;
        color: #FFFFFF !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.25) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94A3B8 !important;
    }

    /* Sidebar Patient Card */
    .patient-card {
        background: #161F30;
        border: 1px solid #283548;
        border-radius: 10px;
        padding: 14px;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .patient-card-title {
        color: #38BDF8 !important;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }
    .patient-card-detail {
        color: #CBD5E1 !important;
        font-size: 0.83rem;
        line-height: 1.45;
    }

    /* Metrics Chip */
    .metric-pill {
        display: inline-block;
        background-color: #1E293B;
        color: #94A3B8 !important;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        margin-top: 8px;
        margin-right: 6px;
        border: 1px solid #334155;
    }

    /* General Input & Select Styling */
    .stTextInput input, .stSelectbox select {
        background-color: #161F30 !important;
        color: #FFFFFF !important;
        border: 1px solid #334155 !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Model & Pipeline Loaders (Shared Memory Singleton)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading DR. MIGI Brain (Qwen 2.5)...")
def get_inference_engine():
    """Loads the core standalone LLM engine once into memory."""
    return DrMigiEngine()


@st.cache_resource(show_spinner="Loading DR. MIGI RAG Pipeline...")
def get_rag_pipeline():
    """Loads the RAG components reusing the existing LLM engine to prevent duplicate RAM usage."""
    shared_engine = get_inference_engine()
    return DrMigiRAGPipeline(config_path="configs/rag_config.json", engine=shared_engine)


# -----------------------------------------------------------------------------
# Patient Profiles Dataset
# -----------------------------------------------------------------------------
PATIENT_PROFILES = {
    "P001": {
        "name": "John Doe",
        "age": 52,
        "gender": "Male",
        "conditions": "Type 2 Diabetes, Hypertension",
        "desc": "Diabetes Trajectory (Pre-diabetes to T2DM over 4 years)",
        "file": "datasets/patients/P001_john_doe.json"
    },
    "P002": {
        "name": "Priya Sharma",
        "age": 45,
        "gender": "Female",
        "conditions": "Hypothyroidism, Dyslipidaemia, Post-NSTEMI",
        "desc": "Cardiac Trajectory (Dyslipidaemia to Acute NSTEMI)",
        "file": "datasets/patients/P002_priya_sharma.json"
    },
    "P003": {
        "name": "Ramesh Iyer",
        "age": 63,
        "gender": "Male",
        "conditions": "COPD GOLD Stage III, T2DM",
        "desc": "Pulmonary Trajectory (COPD + Diabetes to ICU Admission)",
        "file": "datasets/patients/P003_ramesh_iyer.json"
    }
}


# -----------------------------------------------------------------------------
# Sidebar: Patient Selection with Search & Clear Options
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 👤 Patient Selection")

    # Search bar for filtering patients
    search_term = st.text_input("🔍 Search Patient:", placeholder="Type name, ID, or condition...").strip().lower()

    # Filter patient list according to search term
    patient_options = {"None": "None (General Clinical Chat — No Patient RAG)"}
    for pid, pdata in PATIENT_PROFILES.items():
        search_blob = f"{pid} {pdata['name']} {pdata['conditions']} {pdata['desc']}".lower()
        if not search_term or search_term in search_blob:
            patient_options[pid] = f"{pid} — {pdata['name']} ({pdata['conditions'].split(',')[0]})"

    selected_patient_key = st.selectbox(
        "Active Patient Profile:",
        options=list(patient_options.keys()),
        format_func=lambda k: patient_options[k]
    )

    # Patient Summary Card (if patient selected)
    if selected_patient_key != "None":
        p_info = PATIENT_PROFILES[selected_patient_key]
        st.markdown(f"""
        <div class="patient-card">
            <div class="patient-card-title">📋 {p_info['name']} ({selected_patient_key})</div>
            <div class="patient-card-detail">
                <b>Age/Gender:</b> {p_info['age']} {p_info['gender']}<br>
                <b>Conditions:</b> {p_info['conditions']}<br>
                <b>Trajectory:</b> {p_info['desc']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("💡 Chatting in general mode. The assistant will answer medical questions using standard clinical reasoning.")

    st.markdown("---")

    # Chat History Reset
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# Main Screen: Minimalist DR. MIGI Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="migi-header">
    <div class="migi-title">🩺 DR. MIGI</div>
    <div class="migi-subtitle">AI Clinical Intelligence & Healthcare Companion</div>
</div>
""", unsafe_allow_html=True)

# Active Mode Indicator Badge
if selected_patient_key == "None":
    st.markdown("""
    <div class="status-badge-container">
        <span class="status-badge badge-general">💬 General Clinical Chat (Direct LLM Reasoning — No RAG)</span>
    </div>
    """, unsafe_allow_html=True)
else:
    p_info = PATIENT_PROFILES[selected_patient_key]
    st.markdown(f"""
    <div class="status-badge-container">
        <span class="status-badge badge-patient">🧬 Grounded Patient Analysis: {p_info['name']} ({selected_patient_key})</span>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Chat Message State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello, I am **DR. MIGI**, your clinical reasoning assistant. How can I assist you with clinical analysis, medical inquiries, or patient evaluations today?"
        }
    ]


# -----------------------------------------------------------------------------
# Display Chat History
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metrics" in msg and msg["metrics"]:
            m = msg["metrics"]
            st.markdown(f"""
            <span class="metric-pill">⚡ {m.get('tokens_per_second', 0):.1f} tok/s</span>
            <span class="metric-pill">🔢 {m.get('output_tokens_count', 0)} tokens</span>
            <span class="metric-pill">⏱️ {m.get('duration_seconds', 0):.2f}s</span>
            """, unsafe_allow_html=True)

        if "retrieved_context" in msg and msg["retrieved_context"]:
            with st.expander("📑 Retrieved Patient Records & Evidence"):
                st.text(msg["retrieved_context"])


# -----------------------------------------------------------------------------
# Chat Input & Response Generation
# -----------------------------------------------------------------------------
user_input = st.chat_input("Ask DR. MIGI a clinical question...")

if user_input:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        try:
            if selected_patient_key == "None":
                # Mode 1: Direct LLM Chat (No RAG)
                engine = get_inference_engine()
                with st.spinner("DR. MIGI is reasoning..."):
                    metrics = engine.generate_response(
                        prompt=user_input,
                        max_new_tokens=350,
                        temperature=0.3
                    )
                response_text = metrics["response"]
                retrieved_context = None
            else:
                # Mode 2: Grounded Patient RAG Analysis
                pipeline = get_rag_pipeline()
                with st.spinner(f"Retrieving records and analyzing {selected_patient_key}..."):
                    metrics = pipeline.ask(
                        patient_id=selected_patient_key,
                        question=user_input,
                        max_new_tokens=400,
                        temperature=0.3
                    )
                response_text = metrics["response"]
                retrieved_context = metrics.get("retrieved_context", "")

            # Display response
            response_placeholder.markdown(response_text)

            # Show performance pills
            st.markdown(f"""
            <span class="metric-pill">⚡ {metrics.get('tokens_per_second', 0):.1f} tok/s</span>
            <span class="metric-pill">🔢 {metrics.get('output_tokens_count', 0)} tokens</span>
            <span class="metric-pill">⏱️ {metrics.get('duration_seconds', 0):.2f}s</span>
            """, unsafe_allow_html=True)

            # Show retrieved evidence if in RAG mode
            if retrieved_context:
                with st.expander("📑 Retrieved Patient Records & Evidence"):
                    st.text(retrieved_context)

            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "metrics": {
                    "tokens_per_second": metrics.get("tokens_per_second", 0),
                    "output_tokens_count": metrics.get("output_tokens_count", 0),
                    "duration_seconds": metrics.get("duration_seconds", 0)
                },
                "retrieved_context": retrieved_context
            })

        except Exception as e:
            st.error(f"Error generating clinical analysis: {e}")
