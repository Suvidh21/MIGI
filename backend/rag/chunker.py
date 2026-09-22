import json
from typing import Any


class PatientRecordChunker:
    """
    Converts a structured patient JSON record into a list of searchable text chunks.

    What it does:
        Splits a patient's full medical history into individual visit-level chunks.
        Each chunk represents one clinical encounter and contains all relevant
        information from that visit: vitals, diagnosis, lab results, and doctor notes.

    Why it is needed:
        A vector database searches over small, semantically coherent units of text.
        Dumping an entire patient record as one document would make retrieval imprecise.
        By chunking at the visit level, each unit captures a complete clinical snapshot,
        allowing the retriever to find the most relevant visits for any given query.

    Why visit-level chunking (not fixed-size character chunking):
        Fixed-size chunking (e.g., 512 characters) would fragment clinical data —
        splitting a lab result from its interpretation, or a diagnosis from its context.
        Visit-level chunking preserves clinical coherence: every chunk is a complete,
        meaningful unit of medical information.
    """

    def chunk_patient(self, patient_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Converts one patient JSON record into a list of visit-level chunks.

        Arguments:
            patient_data: Parsed patient JSON dictionary.

        Returns:
            List of chunk dictionaries, each with:
                - 'text': The full text representation of the visit (for embedding)
                - 'metadata': Structured metadata for filtering (patient_id, date, etc.)
                - 'chunk_id': A unique identifier for this chunk
        """
        chunks = []
        patient_id = patient_data["patient_id"]
        patient_name = patient_data["name"]
        age = patient_data["age"]
        gender = patient_data["gender"]
        blood_group = patient_data.get("blood_group", "Unknown")
        chronic_conditions = patient_data.get("chronic_conditions", [])
        family_history = patient_data.get("family_history", [])

        # Build a patient header that is prepended to every chunk.
        # This ensures that when a chunk is retrieved, the LLM always knows
        # which patient the visit belongs to and their background context.
        patient_header = (
            f"Patient: {patient_name} | ID: {patient_id} | Age: {age} | "
            f"Gender: {gender} | Blood Group: {blood_group}\n"
            f"Chronic Conditions: {', '.join(chronic_conditions) if chronic_conditions else 'None'}\n"
            f"Family History: {', '.join(family_history) if family_history else 'None'}\n"
        )

        for visit in patient_data.get("visits", []):
            visit_text = self._format_visit(patient_header, visit)
            chunk_id = f"{patient_id}_{visit['visit_id']}"

            chunks.append({
                "chunk_id": chunk_id,
                "text": visit_text,
                "metadata": {
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "visit_id": visit["visit_id"],
                    "date": visit["date"],
                    "diagnosis": visit.get("diagnosis", ""),
                    "chief_complaint": visit.get("chief_complaint", ""),
                }
            })

        return chunks

    def _format_visit(self, patient_header: str, visit: dict[str, Any]) -> str:
        """
        Converts a single visit dictionary into a plain-text paragraph
        optimised for semantic embedding.

        What it does:
            Flattens nested JSON structures (vitals, lab results, medications)
            into readable sentences that an embedding model can understand.
        """
        lines = [patient_header]
        lines.append(f"Visit Date: {visit.get('date', 'Unknown')}")
        lines.append(f"Chief Complaint: {visit.get('chief_complaint', 'Not recorded')}")

        # Vitals
        vitals = visit.get("vitals", {})
        if vitals:
            vitals_parts = []
            for key, value in vitals.items():
                readable_key = key.replace("_", " ").title()
                vitals_parts.append(f"{readable_key}: {value}")
            lines.append("Vitals: " + " | ".join(vitals_parts))

        # Diagnosis
        lines.append(f"Diagnosis: {visit.get('diagnosis', 'Not recorded')}")

        # Medications
        medications = visit.get("medications", [])
        if medications:
            lines.append("Medications: " + ", ".join(medications))
        else:
            lines.append("Medications: None prescribed")

        # Lab Results
        lab_results = visit.get("lab_results", {})
        if lab_results:
            lab_parts = []
            for key, value in lab_results.items():
                readable_key = key.replace("_", " ")
                lab_parts.append(f"{readable_key}: {value}")
            lines.append("Lab Results: " + " | ".join(lab_parts))

        # Doctor Notes — most semantically rich part of the chunk
        notes = visit.get("doctor_notes", "")
        if notes:
            lines.append(f"Doctor Notes: {notes}")

        return "\n".join(lines)

    def chunk_from_file(self, filepath: str) -> list[dict[str, Any]]:
        """
        Convenience method: loads a patient JSON file and returns its chunks.

        Arguments:
            filepath: Absolute or relative path to a patient JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            patient_data = json.load(f)
        return self.chunk_patient(patient_data)
