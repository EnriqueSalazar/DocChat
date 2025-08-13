import time
import unittest
from pathlib import Path
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from docchat.rag_pipeline import RAGPipeline
from docchat.config import Config

TEST_QUESTION = "relation between habsburg and utrecht"
MIN_EXPECTED_KEYWORD = "habsburg"
CONTEXT_KEYWORD = "Utrecht"  # Ensure case-insensitive presence

class TestRAGPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure docs exist (using existing test_doc.md content already in repo)
        assert Path('docs/test_doc.md').exists(), "docs/test_doc.md must exist for this test"
        cfg = Config.load("config.yaml")
        # Use small generation budget for performance test
        cls.pipeline = RAGPipeline(
            embedding_model=cfg.embedding_model,
            chroma_persist_dir=str(cfg.vectorstore_path),
            llm_model_path=cfg.llm_model_name,
            llm_max_new_tokens=128,
            cpu_threads=cfg.cpu_threads,
            enable_gpu=False,
        )
        # Minimal ingestion: if vectorstore empty user should have ingested separately

    def test_answer_latency_and_content(self):
        start = time.time()
        answer, sources = self.pipeline.ask(TEST_QUESTION, top_k=4)
        latency = time.time() - start
        # Assertions
        self.assertLess(latency, 30.0, f"Answer latency too high: {latency:.2f}s")
        self.assertTrue(MIN_EXPECTED_KEYWORD in answer.lower(), f"Answer missing keyword '{MIN_EXPECTED_KEYWORD}': {answer}")
        # Ensure answer references Utrecht indirectly (context retrieval sanity)
        self.assertIn("utrecht", ' '.join(s.lower() for s in sources) + ' ' + answer.lower())
        print(f"Latency: {latency:.2f}s | Answer: {answer} | Sources: {sources}")

if __name__ == '__main__':
    unittest.main()
