# =======================================================================
# DR. MIGI - PROMPT TEMPLATES AND PERSONALITY DEFINITIONS
# =======================================================================

# This file stores the clinical persona and message template generators.
# Keep in mind the future fine-tuning stage: we want training samples to match
# the system prompt and conversation formatting defined here.

# What it does:
#   Defines the role-play boundaries and clinical safety constraints for the AI.
# Why it is needed:
#   Raw models lack safety instructions and would answer query inputs standardly
#   or hallucinate autonomous medical treatment guidelines. A strict system
#   prompt forces Qwen-Instruct to act as an assistant companion.
# Best practices:
#   Clearly declare that the AI does NOT issue primary diagnoses and serves as a tool
#   for clinical decision support and educational reference.
DR_MIGI_SYSTEM_PROMPT = (
    "You are DR. MIGI, a clinical reasoning assistant and patient education companion.\n"
    "Your objective is to provide high-quality medical knowledge retrieval, clinical reasoning assistance, "
    "and health advice.\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. You are NOT a doctor; you are a reasoning assistant. Never issue direct prescriptions or diagnostic statements.\n"
    "2. Always prefix diagnostic ideas with standard clinical qualifiers (e.g., 'Differential diagnoses to consider include...').\n"
    "3. Keep your tone objective, professional, and concise.\n"
    "4. If a scenario is a medical emergency, direct the user immediately to urgent care or emergency services."
)

# =======================================================================
# PHASE 2 — RAG-GROUNDED SYSTEM PROMPT
# =======================================================================

# What it does:
#   Extends the base clinical persona with retrieved patient records injected
#   as grounded evidence. The LLM is instructed to reason over the retrieved
#   context rather than relying solely on pretrained medical knowledge.
# Why it is needed:
#   Without grounding, the LLM answers generically for any patient.
#   With retrieved context injected here, the model reasons over actual,
#   specific patient history — making responses patient-specific and
#   clinically meaningful rather than generic.
# Best practices:
#   Clearly delimit the retrieved context with markers (=== ... ===) so the
#   LLM understands where grounded evidence starts and ends.
DR_MIGI_RAG_SYSTEM_PROMPT_TEMPLATE = (
    "You are DR. MIGI, a clinical reasoning assistant and longitudinal patient analyst.\n"
    "Your role is to analyse patient history, identify clinical trends, and provide "
    "evidence-grounded insights to assist clinicians in decision-making.\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. You are NOT a doctor. You are a reasoning assistant. Never issue prescriptions or definitive diagnoses.\n"
    "2. Ground every observation in the retrieved patient records below. Do not speculate beyond the evidence.\n"
    "3. When identifying trends, cite specific visit dates and values from the records.\n"
    "4. Use clinical qualifiers: 'The data suggests...', 'Based on the retrieved records...', 'A trend is evident in...'\n"
    "5. If a scenario is a medical emergency, direct the clinician to urgent care immediately.\n\n"
    "=== RETRIEVED PATIENT RECORDS (Grounded Evidence) ===\n"
    "{patient_context}\n"
    "=== END OF RETRIEVED RECORDS ===\n\n"
    "Analyse the above records carefully and respond to the clinician's question "
    "with a structured, evidence-based clinical assessment."
)


def build_chat_messages(user_content: str, system_prompt: str = DR_MIGI_SYSTEM_PROMPT) -> list[dict[str, str]]:
    """
    Constructs a standard structured dialogue list for Qwen's chat template.
    Used by Phase 1 (DrMigiEngine) for direct LLM queries without retrieval.

    Arguments:
        user_content: The query string or patient question.
        system_prompt: System guidelines defining the character.

    Returns:
        A list of chat messages formatted for Hugging Face tokenizer templates.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def build_rag_messages(user_query: str, patient_context: str) -> list[dict[str, str]]:
    """
    Constructs a RAG-grounded dialogue list with retrieved patient context injected.
    Used by Phase 2 (DrMigiRAGPipeline) for patient-specific clinical analysis.

    What it does:
        Fills the RAG system prompt template with the retrieved patient records,
        then wraps the clinician's query as the user message. The resulting
        message list is passed to Qwen's apply_chat_template(), which formats
        it into the model's expected input sequence.

    Why it is needed:
        This is where grounding happens. The system prompt now contains real,
        specific patient history — so when the LLM generates its response,
        it reasons over retrieved evidence rather than generic medical knowledge.

    Arguments:
        user_query: The clinician's clinical question about the patient.
        patient_context: The formatted retrieved records string from MigiRetriever.

    Returns:
        A list of chat messages formatted for Hugging Face tokenizer templates,
        with patient context injected into the system role.
    """
    grounded_system_prompt = DR_MIGI_RAG_SYSTEM_PROMPT_TEMPLATE.format(
        patient_context=patient_context
    )
    return [
        {"role": "system", "content": grounded_system_prompt},
        {"role": "user", "content": user_query}
    ]
