from backend.inference.engine import DrMigiEngine
from backend.rag.retriever import MigiRetriever
from backend.rag.embedder import MigiEmbedder
from backend.rag.vector_store import MigiVectorStore
from backend.prompts.templates import build_rag_messages


class DrMigiRAGPipeline:
    """
    The complete RAG orchestration pipeline for DR. MIGI.

    What it does:
        Combines the retrieval layer (MigiRetriever) with the LLM inference layer
        (DrMigiEngine) into a single unified interface. When a doctor asks a question
        about a patient, this pipeline:
            1. Retrieves the most relevant patient history from the vector database
            2. Injects that history into the clinical prompt as grounded context
            3. Passes the enriched prompt to Qwen 2.5 for response generation
            4. Returns the grounded response alongside retrieved context and metrics

    Why it is needed:
        Without RAG, the LLM answers from general medical training only — it has
        no knowledge of any specific patient. With RAG, the LLM reasons over real,
        retrieved patient data. The response is no longer a generic medical answer;
        it is a patient-specific clinical analysis grounded in actual history.

    How it differs from Phase 1 (DrMigiEngine alone):
        Phase 1: query → Qwen 2.5 → generic answer
        Phase 2: query → retrieve patient records → inject context → Qwen 2.5 → grounded answer

    The DrMigiEngine (Phase 1) is unchanged. RAG wraps around it.
    """

    def __init__(
        self,
        config_path: str = "configs/rag_config.json",
        model_name: str | None = None,
        engine: DrMigiEngine | None = None
    ):
        """
        Initialises the full pipeline: embedder, vector store, retriever, and LLM engine.

        Arguments:
            config_path: Path to rag_config.json.
            model_name: Optional override for the LLM model (defaults to config).
            engine: Optional pre-loaded DrMigiEngine instance to share memory.
        """
        print("=" * 60)
        print("Initialising DR. MIGI RAG Pipeline...")
        print("=" * 60)

        # Phase 1 — LLM inference engine (reused or loaded once)
        if engine is not None:
            print("\n[1/3] Reusing shared LLM inference engine...")
            self.engine = engine
        else:
            print("\n[1/3] Loading LLM inference engine (Qwen 2.5)...")
            self.engine = DrMigiEngine(model_name=model_name)

        # Phase 2 — Embedding model
        print("\n[2/3] Loading embedding model (BAAI/bge-small-en-v1.5)...")
        self.embedder = MigiEmbedder(config_path=config_path)

        # Phase 2 — Vector store
        print("\n[3/3] Connecting to ChromaDB vector store...")
        self.vector_store = MigiVectorStore(embedder=self.embedder, config_path=config_path)

        # Phase 2 — Retriever
        self.retriever = MigiRetriever(vector_store=self.vector_store, config_path=config_path)

        print("\n[OK] DR. MIGI RAG Pipeline ready.\n")

    def ask(
        self,
        patient_id: str,
        question: str,
        use_full_timeline: bool = False,
        **generation_kwargs
    ) -> dict:
        """
        The main interface: given a patient ID and a clinical question,
        returns a grounded response based on retrieved patient history.

        Arguments:
            patient_id: The patient to query (e.g., 'P001').
            question: The clinical question from the doctor.
            use_full_timeline: If True, retrieves the patient's complete history
                               instead of semantic search. Useful for summary requests.
            generation_kwargs: Optional overrides for LLM generation parameters.

        Returns:
            Dictionary containing:
                - 'response': The LLM's grounded clinical response
                - 'retrieved_context': The patient history injected into the prompt
                - 'input_tokens_count': Number of tokens in the full prompt
                - 'output_tokens_count': Number of tokens generated
                - 'tokens_per_second': Generation speed
                - 'duration_seconds': Total generation time
                - 'device': Device used for inference (cpu/cuda)
                - 'patient_id': The queried patient ID
                - 'question': The original clinical question
        """
        # Step 1: Retrieve relevant patient history
        if use_full_timeline:
            retrieved_context = self.retriever.retrieve_full_timeline(patient_id)
        else:
            retrieved_context = self.retriever.retrieve(query=question, patient_id=patient_id)

        # Step 2: Build the RAG-enhanced system prompt with retrieved context injected
        # The system prompt now includes the patient history as grounded evidence
        messages = build_rag_messages(
            user_query=question,
            patient_context=retrieved_context
        )

        # Step 3: Format messages into Qwen chat template and generate response
        prompt_text = self.engine.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Step 4: Run inference through Phase 1 engine
        # Note: We call generate_response with the fully assembled RAG prompt
        metrics = self.engine.generate_response(
            prompt=question,
            system_prompt=self._build_rag_system_prompt(retrieved_context),
            **generation_kwargs
        )

        # Augment the result with RAG-specific information
        metrics["retrieved_context"] = retrieved_context
        metrics["patient_id"] = patient_id
        metrics["question"] = question

        return metrics

    def _build_rag_system_prompt(self, retrieved_context: str) -> str:
        """
        Constructs the RAG-enhanced system prompt by injecting retrieved patient
        records into the clinical persona prompt.

        What it does:
            Combines the base clinical persona (from Phase 1) with the retrieved
            patient records, creating a grounded context for the LLM to reason over.
        """
        return (
            "You are DR. MIGI, a clinical reasoning assistant and patient education companion.\n"
            "Your objective is to provide high-quality medical analysis grounded in the patient records provided.\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. You are NOT a doctor; you are a reasoning assistant. Never issue direct prescriptions or diagnostic statements.\n"
            "2. Always prefix diagnostic ideas with standard clinical qualifiers (e.g., 'Based on the retrieved records, the data suggests...').\n"
            "3. Ground every observation in the specific retrieved patient data below. Do not speculate beyond the evidence.\n"
            "4. Identify trends and patterns across multiple visits when present.\n"
            "5. If a scenario is a medical emergency, direct the user immediately to urgent care.\n\n"
            "=== RETRIEVED PATIENT RECORDS ===\n"
            f"{retrieved_context}\n"
            "=== END OF RETRIEVED RECORDS ===\n\n"
            "Analyse the above records and respond to the clinician's question with a grounded, evidence-based assessment."
        )

    def ask_stream(self, patient_id: str, question: str, **generation_kwargs):
        """
        Streaming version of ask() — yields response tokens one by one.

        Why it is needed:
            On CPU, generation is slow. Streaming returns tokens immediately as they
            are generated, making the interface feel responsive during long analyses.

        Yields:
            String tokens as they are generated by the LLM.
        """
        retrieved_context = self.retriever.retrieve(query=question, patient_id=patient_id)
        rag_system_prompt = self._build_rag_system_prompt(retrieved_context)

        yield f"[Retrieved {retrieved_context.count('--- Retrieved Record')} record(s) for {patient_id}]\n\n"

        for token in self.engine.generate_stream(
            prompt=question,
            system_prompt=rag_system_prompt,
            **generation_kwargs
        ):
            yield token
