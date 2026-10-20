"""
Comprehensive Functional Verification Test Suite for EuroJobsCenter RAG Voice Agent.
Runs tests without loading the 3.4GB EuroLLM model.
"""

import io
import json
import unittest
import wave
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Import app components
import app


class TestVoiceAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.app.config["TESTING"] = True
        cls.client = app.app.test_client()

    # -------------------------------------------------------------
    # 1. System & Health Endpoints
    # -------------------------------------------------------------
    def test_01_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("device", data)

    def test_02_voice_status_endpoint(self):
        res = self.client.get("/api/voice/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("whisper_loaded", data)
        self.assertIn("piper_loaded", data)
        self.assertIn("device", data)
        self.assertEqual(data["whisper_model"], "openai/whisper-base.en")

    def test_03_reset_endpoint(self):
        res = self.client.post("/reset", json={"session_id": "test_session"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["session_id"], "test_session")

    # -------------------------------------------------------------
    # 2. Text Cleaning for Speech & Predictable Truncation
    # -------------------------------------------------------------
    def test_04_clean_text_for_speech(self):
        # Test markdown stripping, citation bracket removal, and punctuation normalization
        raw = "[1] section=Visa Steps | file=steps.jsonl\nYou must bring your **passport** [1] and [form](https://gov.de)."
        cleaned, limited = app.clean_text_for_speech(raw)
        self.assertNotIn("[1]", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("passport", cleaned)
        self.assertIn("form", cleaned)
        self.assertFalse(limited)

        # Empty string handling
        self.assertEqual(app.clean_text_for_speech("")[0], "")
        self.assertEqual(app.clean_text_for_speech("   ")[0], "")

        # Long text with predictable sentence completion notice
        long_text = "This is a sentence about German immigration regulations. " * 30
        truncated, limited = app.clean_text_for_speech(long_text, max_chars=120)
        self.assertTrue(limited)
        self.assertIn("For complete details, please see the full answer on your screen.", truncated)

    # -------------------------------------------------------------
    # 3. Text-to-Speech (Piper)
    # -------------------------------------------------------------
    def test_05_synthesize_valid_text(self):
        res = self.client.post("/api/voice/synthesize", json={"text": "Welcome to Germany! Safe travels."})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "audio/wav")
        self.assertGreater(len(res.data), 1000)
        self.assertIn("X-Text-Limited", res.headers)
        self.assertEqual(res.headers["X-Text-Limited"], "false")

        # Verify it is a valid playable WAV
        with wave.open(io.BytesIO(res.data), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertGreater(wf.getnframes(), 0)

    def test_06_synthesize_invalid_inputs(self):
        # Empty text
        res = self.client.post("/api/voice/synthesize", json={"text": ""})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

        # Whitespace/punctuation only
        res = self.client.post("/api/voice/synthesize", json={"text": "[1] *** [2]"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    # -------------------------------------------------------------
    # 4. Speech-to-Text (Whisper) & WAV Validation
    # -------------------------------------------------------------
    def _create_wav(self, duration_sec=1.0, frequency=0.0):
        """Helper to create dummy 16kHz mono WAV bytes (silence or tone)."""
        sr = 16000
        n_samples = int(sr * duration_sec)
        if frequency > 0:
            t = np.linspace(0, duration_sec, n_samples, endpoint=False)
            audio = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        else:
            audio = np.zeros(n_samples, dtype=np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    def test_07_transcribe_missing_or_corrupt_audio(self):
        # No audio data
        res = self.client.post("/api/voice/transcribe", data=b"")
        self.assertEqual(res.status_code, 400)

        # Corrupt non-WAV bytes
        res = self.client.post("/api/voice/transcribe", data=b"NOT_A_WAV_FILE_HEADER_LONGER_THAN_44_BYTES_PADDING_1234567890")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid audio container format", res.get_json()["error"])

    def test_08_transcribe_silence_detection(self):
        # Send pure silence WAV
        silent_wav = self._create_wav(duration_sec=1.5, frequency=0.0)
        res = self.client.post("/api/voice/transcribe", data=silent_wav, content_type="audio/wav")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["text"], "")
        self.assertTrue(data["silence"])
        self.assertGreater(data["duration"], 1.0)

    def test_09_transcribe_real_spoken_speech(self):
        # Synthesize real speech via Piper, then transcribe via Whisper
        synth_res = self.client.post("/api/voice/synthesize", json={"text": "How do I apply for a visa?"})
        self.assertEqual(synth_res.status_code, 200)
        speech_wav = synth_res.data

        # Send to transcribe endpoint
        res = self.client.post(
            "/api/voice/transcribe",
            data={"audio": (io.BytesIO(speech_wav), "speech.wav")},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertFalse(data["silence"])
        self.assertIn("visa", data["text"].lower())

    # -------------------------------------------------------------
    # 5. Text-Chat Regression & Mocked /ask Pipeline
    # -------------------------------------------------------------
    def test_10_ask_endpoint_empty_question(self):
        res = self.client.post("/ask", json={"question": ""})
        self.assertEqual(res.status_code, 400)

    @patch("app.generate_answer")
    def test_11_ask_endpoint_success(self, mock_generate):
        mock_generate.return_value = "You must register your address within two weeks."

        res = self.client.post("/ask", json={
            "session_id": "test_session_2",
            "question": "What is the deadline for address registration?",
            "debug": True
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("answer", data)
        self.assertEqual(data["answer"], "You must register your address within two weeks.")
        self.assertIn("sources", data)
        self.assertIn("meta", data)
        self.assertIn("debug", data)

    # -------------------------------------------------------------
    # 6. Simulated Frontend Voice Flow Sequence
    # -------------------------------------------------------------
    @patch("app.generate_answer")
    def test_12_simulated_full_voice_turn(self, mock_generate):
        mock_generate.return_value = "You must register your address at the Bürgeramt."

        # Step 1: User speaks question -> synthesized to WAV
        speech_wav = self.client.post(
            "/api/voice/synthesize",
            json={"text": "Where do I register my address?"}
        ).data

        # Step 2: Frontend sends WAV to /api/voice/transcribe
        transcribe_res = self.client.post(
            "/api/voice/transcribe",
            data={"audio": (io.BytesIO(speech_wav), "speech.wav")},
            content_type="multipart/form-data"
        )
        self.assertEqual(transcribe_res.status_code, 200)
        transcript = transcribe_res.get_json()["text"]
        self.assertIn("address", transcript.lower())

        # Step 3: Frontend calls /ask with transcribed text
        ask_res = self.client.post("/ask", json={
            "session_id": "voice_user_1",
            "question": transcript
        })
        self.assertEqual(ask_res.status_code, 200)
        answer = ask_res.get_json()["answer"]
        self.assertEqual(answer, "You must register your address at the Bürgeramt.")

        # Step 4: Frontend calls /api/voice/synthesize with answer
        tts_res = self.client.post("/api/voice/synthesize", json={"text": answer})
        self.assertEqual(tts_res.status_code, 200)
        self.assertEqual(tts_res.mimetype, "audio/wav")
        self.assertGreater(len(tts_res.data), 1000)

    # -------------------------------------------------------------
    # 7. Device Resolution & Portability Validation
    # -------------------------------------------------------------
    def test_13_device_resolution_validation(self):
        # Explicit CPU
        dev, dtype = app.resolve_device("cpu")
        self.assertEqual(dev, "cpu")
        self.assertEqual(dtype, torch.bfloat16)

        # Invalid device string
        with self.assertRaises(ValueError):
            app.resolve_device("tpu_invalid_device")

        # Unsupported CUDA on machine without GPU
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                app.resolve_device("cuda")

    # -------------------------------------------------------------
    # 8. Strict Audio Request Validation (Payload Size, RIFF Header)
    # -------------------------------------------------------------
    def test_14_audio_validation_strict_riff(self):
        # Less than 44 bytes (truncated header)
        res = self.client.post("/api/voice/transcribe", data=b"RIFFshort")
        self.assertEqual(res.status_code, 400)
        self.assertIn("too short", res.get_json()["error"])

        # Fake 50-byte header without RIFF/WAVE
        fake_data = b"0" * 50
        res = self.client.post("/api/voice/transcribe", data=fake_data)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid audio container", res.get_json()["error"])

        # Oversized audio payload test
        oversized_data = b"RIFF" + b"WAVE" + b"0" * (app.MAX_AUDIO_BYTES + 10)
        res = self.client.post("/api/voice/transcribe", data=oversized_data)
        self.assertEqual(res.status_code, 413)

    # -------------------------------------------------------------
    # 9. Concurrency & Thread Locks
    # -------------------------------------------------------------
    def test_15_thread_locks_guard(self):
        self.assertTrue(hasattr(app, "_whisper_lock"))
        self.assertTrue(hasattr(app, "_piper_lock"))
        self.assertTrue(hasattr(app, "_inference_lock"))

        # Verify lock can be acquired and released cleanly
        with app._inference_lock:
            self.assertTrue(app._inference_lock.locked())
        self.assertFalse(app._inference_lock.locked())

    # -------------------------------------------------------------
    # 10. Bounded Pre-Speech Silence Buffer Verification
    # -------------------------------------------------------------
    def test_16_bounded_silence_buffer_simulation(self):
        # Simulates the frontend pre-speech ring buffer logic
        chunks = []
        limit = 4
        # Add 200 silent audio chunks
        for i in range(200):
            chunks.append(np.zeros(4096, dtype=np.float32))
            if len(chunks) > limit:
                chunks.pop(0)

        self.assertEqual(len(chunks), limit)
        total_samples = sum(len(c) for c in chunks)
        self.assertEqual(total_samples, 4 * 4096)


if __name__ == "__main__":
    unittest.main()
