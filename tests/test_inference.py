import os
import sys
import unittest

# Configure Hugging Face cache redirection for tests
os.environ["HF_HOME"] = "I:/suvidh/MIGI-main/models/cache"

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference.engine import DrMigiEngine

class TestDrMigiEngine(unittest.TestCase):
    """
    Test suite for validating the DR. MIGI core reasoning engine.
    
    Why it is needed:
        Verifies that refactorings and code edits do not break model loading,
        chat formatting templates, or inference outputs.
    """
    @classmethod
    def setUpClass(cls):
        """
        Loads the engine once for all tests to conserve CPU and RAM load times.
        """
        print("\nLoading DrMigiEngine for test suite (this might take a few moments)...")
        cls.engine = DrMigiEngine()

    def test_engine_initialization(self):
        """
        Validates model and tokenizer instances were instantiated properly.
        """
        self.assertIsNotNone(self.engine.model)
        self.assertIsNotNone(self.engine.tokenizer)
        print("[PASS] Test passed: Model and tokenizer initialized.")

    def test_synchronous_generation(self):
        """
        Validates synchronous generation returns expected keys, datatypes, and content.
        """
        prompt = "Hello, respond with exactly one word: 'Affirmative'."
        metrics = self.engine.generate_response(prompt, max_new_tokens=10, temperature=0.1)
        
        self.assertIn("response", metrics)
        self.assertIn("input_tokens_count", metrics)
        self.assertIn("output_tokens_count", metrics)
        self.assertIn("tokens_per_second", metrics)
        self.assertIn("device", metrics)
        
        self.assertIsInstance(metrics["response"], str)
        self.assertGreater(len(metrics["response"]), 0)
        self.assertGreaterEqual(metrics["input_tokens_count"], 1)
        self.assertGreaterEqual(metrics["output_tokens_count"], 1)
        
        print(f"[PASS] Test passed: Response generated: '{metrics['response']}'")
        print(f"  Prefill count: {metrics['input_tokens_count']} tokens | Output count: {metrics['output_tokens_count']} tokens")

    def test_streaming_generation(self):
        """
        Validates streaming generator returns non-empty string chunks.
        """
        prompt = "Explain in one sentence what medicine is."
        stream = self.engine.generate_stream(prompt, max_new_tokens=15, temperature=0.1)
        
        chunks = []
        for chunk in stream:
            self.assertIsInstance(chunk, str)
            chunks.append(chunk)
            
        full_streamed_response = "".join(chunks)
        self.assertGreater(len(full_streamed_response), 0)
        print(f"[PASS] Test passed: Stream completed. Response length: {len(full_streamed_response)} chars.")

if __name__ == "__main__":
    unittest.main()
