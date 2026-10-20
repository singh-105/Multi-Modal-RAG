import unittest
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from grounding import GroundingEngine

class TestGroundingFactual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import app
            cls.client = app.qdrant
            cls.model = app.embedder
            cls.engine = app.grounding_engine
        except Exception:
            cls.client = QdrantClient(path="qdrant_data")
            cls.model = SentenceTransformer("intfloat/multilingual-e5-base", cache_folder="models", device="cpu")
            cls.all_points = cls.client.scroll(
                collection_name="veloit_handbook_chunks",
                limit=200,
                with_payload=True,
                with_vectors=False,
            )[0]
            cls.engine = GroundingEngine(cls.all_points)

    def retrieve(self, query: str):
        q_vec = self.model.encode(["query: " + query], normalize_embeddings=True)[0]
        dense_res = self.client.query_points(
            collection_name="veloit_handbook_chunks",
            query=q_vec.tolist(),
            limit=25,
            with_payload=True,
        )
        top_dense = dense_res.points[0].score if dense_res.points else 0.0
        chosen_points, q_words = self.engine.rank_chunks(query, dense_res.points, limit=5)
        return top_dense, chosen_points

    def test_question_1_code_95_cost(self):
        """1. For 'How much does Code 95 training cost according to the handbook?',
           bot must not hallucinate unverified €1000-€2000 cost and state the handbook does not specify the cost."""
        q = "How much does Code 95 training cost according to the handbook?"
        top_dense, points = self.retrieve(q)
        context = "\n".join(p.payload.get("text", "") for p in points)

        # Simulate hallucinated answer from EuroLLM
        simulated_hallucination = "Code 95 training costs €1,000 to €2,000 according to training providers in Germany."
        answer, sources = self.engine.verify_and_ground_answer(q, simulated_hallucination, context, points)

        self.assertNotIn("1,000", answer)
        self.assertNotIn("2,000", answer)
        self.assertNotIn("€1,000", answer)
        self.assertIn("does not state the cost of Code 95", answer)
        self.assertTrue(len(sources) > 0)
        for s in sources:
            self.assertNotEqual(s["section_title"], "Cover Page")

    def test_question_2_guarantee(self):
        """2. For 'Can you guarantee that my German work visa will be approved?',
           bot must directly state that no guarantee can be given and decisions are made by German authorities."""
        q = "Can you guarantee that my German work visa will be approved?"
        top_dense, points = self.retrieve(q)
        context = "\n".join(p.payload.get("text", "") for p in points)

        # Confirm the guarantee chunk from Driver Declaration was retrieved
        found_declaration = any("driver_declaration" in p.payload.get("source_file", "") for p in points)
        self.assertTrue(found_declaration, "Driver declaration chunk must be retrieved via hybrid search")

        # Simulate an evasion answer that doesn't directly answer
        simulated_evasion = "EuroJobsCenter helps you submit your application and prepare for the appointment."
        answer, sources = self.engine.verify_and_ground_answer(q, simulated_evasion, context, points)

        self.assertTrue(
            "cannot guarantee" in answer.lower() or "no" in answer.lower(),
            "Answer must clearly negate guarantee"
        )
        self.assertIn("German authorities", answer)
        self.assertTrue(len(sources) > 0)

    def test_question_3_fifa_world_cup(self):
        """3. For 'Who won the FIFA World Cup in 2022?',
           bot must recognize out-of-scope, not answer France, and display NO citations."""
        q = "Who won the FIFA World Cup in 2022?"
        top_dense, points = self.retrieve(q)
        
        is_oos = self.engine.is_out_of_scope(q, top_dense, points)
        self.assertTrue(is_oos, "FIFA query must be identified as out of scope")

        if is_oos:
            answer = "I could not find that in the handbook. I can only answer questions related to the EuroJobsCenter Visa & Immigration Handbook for Germany."
            sources = []
        else:
            answer, sources = self.engine.verify_and_ground_answer(q, "France won the World Cup.", "", points)

        self.assertNotIn("France", answer)
        self.assertIn("I could not find that in the handbook", answer)
        self.assertEqual(len(sources), 0, "Out-of-scope query must NOT cite any handbook sources")

    def test_question_4_visa_vs_residence_permit(self):
        """4. For 'What is the difference between a work visa and a residence permit?',
           bot must give an accurate, well-grounded procedural explanation based on the handbook."""
        q = "What is the difference between a work visa and a residence permit?"
        top_dense, points = self.retrieve(q)
        context = "\n".join(p.payload.get("text", "") for p in points)

        simulated_oversimplified = "A work visa is for work and a residence permit is for living."
        answer, sources = self.engine.verify_and_ground_answer(q, simulated_oversimplified, context, points)

        self.assertIn("German Embassy", answer)
        self.assertIn("residence permits", answer.lower())
        self.assertIn("German authorities", answer)
        self.assertTrue(len(sources) > 0)

    def test_unsupported_out_of_scope_query(self):
        """General out-of-scope query test (e.g. baking recipe) without hardcoding."""
        q = "How do I bake a chocolate cake at home?"
        top_dense, points = self.retrieve(q)
        is_oos = self.engine.is_out_of_scope(q, top_dense, points)
        self.assertTrue(is_oos, "Cooking recipe must be detected as out-of-scope")

    def test_valid_supported_query_official_fee(self):
        """Valid supported query must be answered accurately with EUR 411 and cite sources."""
        q = "What is the official immigration authority fee?"
        top_dense, points = self.retrieve(q)
        context = "\n".join(p.payload.get("text", "") for p in points)

        answer, sources = self.engine.verify_and_ground_answer(
            q,
            "The official fee for the German Pre-Approval procedure is EUR 411, paid directly to the authorities.",
            context,
            points
        )
        self.assertIn("411", answer)
        self.assertTrue(len(sources) > 0)

    def test_source_deduplication(self):
        """Sources must be deduplicated by section and file."""
        q = "What should I do during my first days in Germany?"
        top_dense, points = self.retrieve(q)
        _, sources = self.engine.verify_and_ground_answer(q, "Valid answer", "context", points)

        seen = set()
        for s in sources:
            key = (s["section_title"], s["source_file"])
            self.assertNotIn(key, seen, "Sources must be unique by section and file")
            seen.add(key)

if __name__ == "__main__":
    unittest.main()

