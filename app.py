import io
import os
import re
import threading
import time
import wave
from collections import defaultdict

import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from grounding import GroundingEngine

QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "veloit_handbook_chunks"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
LLM_NAME = "utter-project/EuroLLM-1.7B-Instruct"
MODEL_CACHE = "models"

WHISPER_MODEL_NAME = "openai/whisper-base.en"
PIPER_VOICE_REPO = "rhasspy/piper-voices"
PIPER_VOICE_FILE = "en/en_US/lessac/low/en_US-lessac-low.onnx"
PIPER_CONFIG_FILE = "en/en_US/lessac/low/en_US-lessac-low.onnx.json"
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 45.0
SILENCE_RMS_THRESHOLD = 0.005
MAX_TTS_CHARS = 1000

MAX_HISTORY_TURNS = 6
MAX_SESSION_CHARS = 6000
MAX_CONTEXT_CHARS = 5000
MIN_RETRIEVAL_SCORE = 0.42

app = Flask(__name__)
chat_memory = defaultdict(list)

_whisper_lock = threading.Lock()
_piper_lock = threading.Lock()
_inference_lock = threading.Lock()


def resolve_device(forced_env: str = None) -> tuple[str, torch.dtype]:
    """Validate and resolve target device and model precision."""
    if forced_env:
        forced = forced_env.strip().lower()
        if forced == "cuda" or forced.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"DEVICE='{forced_env}' requested, but torch.cuda.is_available() is False. "
                    "Ensure an NVIDIA GPU with appropriate CUDA drivers is present."
                )
            return forced, torch.float16
        elif forced == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError(
                    "DEVICE='mps' requested, but torch.backends.mps.is_available() is False. "
                    "Apple Silicon MPS is not supported on this platform."
                )
            return "mps", torch.float16
        elif forced == "cpu":
            return "cpu", torch.bfloat16
        else:
            raise ValueError(
                f"Unsupported DEVICE='{forced_env}'. "
                "Supported options are: 'cpu', 'cuda', 'cuda:<index>', 'mps'."
            )

    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.bfloat16


device, model_dtype = resolve_device(os.environ.get("DEVICE"))
print(f"Using device: {device} (dtype: {model_dtype})")

print("Loading Qdrant...")
qdrant = QdrantClient(path=QDRANT_PATH)

print("Initializing grounding engine...")
try:
    _all_points = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=250,
        with_payload=True,
        with_vectors=False,
    )[0]
except Exception as _e:
    print(f"Warning: could not scroll Qdrant points: {_e}")
    _all_points = []
grounding_engine = GroundingEngine(_all_points)

print("Loading embedder...")
embedder = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=MODEL_CACHE, device=device)

print("Loading EuroLLM tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, cache_dir=MODEL_CACHE)

print("Loading EuroLLM model...")
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    cache_dir=MODEL_CACHE,
    dtype=model_dtype,
)
model.to(device)
model.eval()

SYSTEM_PROMPT = (
    "You are the official EuroJobsCenter Germany Handbook Assistant. "
    "Your answers must be concise, factual, and strictly grounded in the provided handbook context. "
    "Never fabricate prices, costs, fees, or legal guarantees that are not explicitly stated. "
    "If the answer or specific detail requested is not in the context, state: 'I could not find that in the handbook.' "
    "For questions about visa or employment guarantees, state clearly that no guarantee can be given by EuroJobsCenter, as decisions are made solely by German authorities under German law."
)


def estimate_chars(messages):
    total = 0
    for item in messages:
        total += len(item.get("role", "")) + len(item.get("content", ""))
    return total


def prune_session_history(history):
    while len(history) > MAX_HISTORY_TURNS * 2:
        history.pop(0)

    while history and estimate_chars(history) > MAX_SESSION_CHARS:
        history.pop(0)

    return history


def retrieve_chunks(query: str, limit: int = 5):
    query_vec = embedder.encode([f"query: {query}"], normalize_embeddings=True)[0]
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec.tolist(),
        limit=25,
        with_payload=True,
        with_vectors=False,
    )
    top_dense = float(response.points[0].score) if (response.points and response.points[0].score is not None) else 0.0
    chosen_points, _ = grounding_engine.rank_chunks(query, response.points, limit=limit)
    return chosen_points, top_dense


def build_context(points):
    blocks = []
    for i, point in enumerate(points, start=1):
        p = point.payload or {}
        blocks.append(
            f"[{i}] section={p.get('section_title')} | "
            f"subsection={p.get('subsection_title')} | "
            f"file={p.get('source_file')}\n{p.get('text')}"
        )
    return "\n\n".join(blocks)


def build_debug_chunks(points):
    items = []
    for point in points:
        p = point.payload or {}
        items.append(
            {
                "score": round(float(point.score), 4) if point.score is not None else None,
                "chunk_id": p.get("chunk_id"),
                "section_title": p.get("section_title"),
                "subsection_title": p.get("subsection_title"),
                "source_file": p.get("source_file"),
                "preview": (p.get("text") or "")[:280],
            }
        )
    return items


def generate_answer(user_message: str, context: str, history):
    compact_context = context[:MAX_CONTEXT_CHARS]

    prompt = f"""You are answering a question based strictly on the EuroJobsCenter Handbook excerpts below.

Context:
{compact_context}

User question:
{user_message}

Instructions:
- Answer directly, accurately, and concisely.
- Use ONLY facts stated in the context above.
- If a specific detail (such as cost, price, or timeline) is not mentioned in the context, explicitly state that it is not in the handbook.
- Do not invent any numbers, prices, or guarantees.
- If asked about guarantees, explicitly state that visa approval cannot be guaranteed and is decided solely by German authorities under German law.
"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in history[-8:]:
        messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "user", "content": prompt})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "device": device,
            "session_mode": "single-session-in-memory",
        }
    )


@app.post("/reset")
def reset():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default")
    chat_memory[session_id] = []
    return jsonify({"ok": True, "session_id": session_id})


@app.post("/ask")
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "default")
    debug = bool(data.get("debug", False))

    if not question:
        return jsonify({"error": "question is required"}), 400

    history = chat_memory[session_id]
    prune_session_history(history)

    points, top_score = retrieve_chunks(question, limit=5)

    if not points or grounding_engine.is_out_of_scope(question, top_score, points):
        answer = "I could not find that in the handbook. I can only answer questions related to the EuroJobsCenter Visa & Immigration Handbook for Germany."
        sources = []
        context = ""
    else:
        context = build_context(points)
        raw_answer = generate_answer(question, context, history)
        answer, sources = grounding_engine.verify_and_ground_answer(question, raw_answer, context, points)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    prune_session_history(history)

    session_chars = estimate_chars(history)
    session_fill_ratio = min(session_chars / MAX_SESSION_CHARS, 1.0)

    response = {
        "session_id": session_id,
        "answer": answer,
        "sources": sources,
        "meta": {
            "top_score": round(top_score, 4),
            "retrieved_chunks": len(points),
            "session_chars": session_chars,
            "session_char_limit": MAX_SESSION_CHARS,
            "session_fill_ratio": session_fill_ratio,
            "history_messages": len(history),
            "context_chars_used": min(len(context), MAX_CONTEXT_CHARS),
            "context_char_limit": MAX_CONTEXT_CHARS,
        },
    }

    if debug:
        response["debug"] = {
            "retrieved_chunks": build_debug_chunks(points),
            "context_preview": context[:1200],
        }

    return jsonify(response)


_whisper_processor = None
_whisper_model = None
_piper_voice = None


def get_whisper():
    global _whisper_processor, _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                print("Loading Whisper STT model...")
                _whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME, cache_dir=MODEL_CACHE)
                whisper_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
                _whisper_model = WhisperForConditionalGeneration.from_pretrained(
                    WHISPER_MODEL_NAME,
                    cache_dir=MODEL_CACHE,
                    torch_dtype=whisper_dtype,
                ).to(device)
                _whisper_model.eval()
    return _whisper_processor, _whisper_model


def get_piper():
    global _piper_voice
    if _piper_voice is None:
        with _piper_lock:
            if _piper_voice is None:
                from piper import PiperVoice
                from huggingface_hub import hf_hub_download
                print("Loading Piper TTS voice...")
                onnx_rel = os.path.normpath(PIPER_VOICE_FILE)
                json_rel = os.path.normpath(PIPER_CONFIG_FILE)
                onnx_path = os.path.join(MODEL_CACHE, onnx_rel)
                json_path = os.path.join(MODEL_CACHE, json_rel)

                if not (os.path.exists(onnx_path) and os.path.exists(json_path)):
                    hf_hub_download(repo_id=PIPER_VOICE_REPO, filename=PIPER_VOICE_FILE, local_dir=MODEL_CACHE)
                    hf_hub_download(repo_id=PIPER_VOICE_REPO, filename=PIPER_CONFIG_FILE, local_dir=MODEL_CACHE)

                _piper_voice = PiperVoice.load(onnx_path, config_path=json_path)
    return _piper_voice


def _warmup_piper():
    try:
        get_piper()
        print("Piper TTS voice warmed up and ready in memory.")
    except Exception as e:
        print(f"Piper warmup note: {e}")


threading.Thread(target=_warmup_piper, daemon=True).start()


def clean_text_for_speech(text: str, max_chars: int = MAX_TTS_CHARS) -> tuple[str, bool]:
    """Clean text for TTS. Returns (cleaned_text, was_limited)."""
    if not text:
        return "", False
    # Strip context header prefixes like [1] section=... | file=...
    text = re.sub(r"\[\d+\]\s*(section|subsection|file)\s*=[^\n]+", "", text, flags=re.IGNORECASE)
    # Strip bracket citations [1], [2]
    text = re.sub(r"\[\d+\]", "", text)
    # Convert markdown links [Text](url) to Text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Strip markdown symbols
    text = re.sub(r"[*#_`>~]", "", text)
    # Fix spacing before punctuation
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    # Normalize whitespace
    text = " ".join(text.split()).strip()

    was_limited = False
    if len(text) > max_chars:
        was_limited = True
        trimmed = text[:max_chars]
        last_punct = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
        if last_punct > 150:
            text = trimmed[:last_punct + 1] + " For complete details, please see the full answer on your screen."
        else:
            text = trimmed.rstrip() + "... For complete details, please see the full answer on your screen."

    return text, was_limited


@app.post("/api/voice/transcribe")
def voice_transcribe():
    # 1. Retrieve audio payload
    audio_bytes = None
    if "audio" in request.files:
        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()
    elif request.data:
        audio_bytes = request.data

    if not audio_bytes or len(audio_bytes) == 0:
        return jsonify({"error": "No audio data provided"}), 400

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        return jsonify({"error": f"Audio payload exceeds {MAX_AUDIO_BYTES // (1024 * 1024)}MB maximum limit"}), 413

    if len(audio_bytes) < 44:
        return jsonify({"error": "Audio payload is too short to be a valid WAV file"}), 400

    if not (audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"):
        return jsonify({"error": "Invalid audio container format. Expected standard RIFF/WAVE."}), 400

    # 2. Decode WAV
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            framerate = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()

            if framerate <= 0:
                return jsonify({"error": f"Invalid audio framerate: {framerate}"}), 400

            if n_channels < 1 or n_channels > 2:
                return jsonify({"error": f"Unsupported channel count: {n_channels} (mono or stereo required)"}), 400

            duration = n_frames / float(framerate)

            if duration > MAX_AUDIO_SECONDS:
                return jsonify({"error": f"Audio duration ({duration:.1f}s) exceeds max limit of {MAX_AUDIO_SECONDS}s"}), 400

            sampwidth = wf.getsampwidth()
            if sampwidth not in (2, 4):
                return jsonify({"error": f"Unsupported sample width: {sampwidth * 8}-bit (16-bit or 32-bit required)"}), 400

            frames = wf.readframes(n_frames)
            if sampwidth == 2:
                audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_np = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0

            if n_channels > 1:
                audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)

            if framerate != 16000:
                target_length = int(len(audio_np) * 16000 / framerate)
                if target_length > 0:
                    audio_np = np.interp(
                        np.linspace(0, len(audio_np), target_length, endpoint=False),
                        np.arange(len(audio_np)),
                        audio_np,
                    ).astype(np.float32)

    except wave.Error as e:
        return jsonify({"error": f"Invalid or malformed WAV audio: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Audio decode failed: {str(e)}"}), 400

    # 3. Check silence / energy
    if len(audio_np) == 0:
        return jsonify({"text": "", "silence": True, "duration": 0.0})

    rms = float(np.sqrt(np.mean(np.square(audio_np))))
    if rms < SILENCE_RMS_THRESHOLD:
        return jsonify({"text": "", "silence": True, "duration": round(duration, 2), "rms": round(rms, 5)})

    # 4. Transcribe using Whisper
    try:
        with _inference_lock:
            processor, whisper = get_whisper()
            inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
            inputs = {
                k: v.to(device=device, dtype=whisper.dtype) if v.is_floating_point() else v.to(device=device)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                predicted_ids = whisper.generate(inputs["input_features"])

            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        return jsonify({
            "text": transcription,
            "silence": False,
            "duration": round(duration, 2),
            "rms": round(rms, 4),
        })
    except Exception as e:
        return jsonify({"error": f"Transcription inference error: {str(e)}"}), 500


@app.post("/api/voice/synthesize")
def voice_synthesize():
    t_start = time.perf_counter()
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    t_clean_start = time.perf_counter()
    cleaned, was_limited = clean_text_for_speech(text)
    t_clean_ms = (time.perf_counter() - t_clean_start) * 1000

    if not cleaned:
        return jsonify({"error": "No speakable text after cleaning"}), 400

    try:
        with _inference_lock:
            t_load_start = time.perf_counter()
            voice = get_piper()
            t_load_ms = (time.perf_counter() - t_load_start) * 1000

            t_synth_start = time.perf_counter()
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                voice.synthesize_wav(cleaned, wav_file)
            buf.seek(0)
            wav_bytes = buf.read()
            t_synth_ms = (time.perf_counter() - t_synth_start) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000
        print(
            f"[TTS Synth] chars={len(cleaned)} | "
            f"clean={t_clean_ms:.2f}ms | load={t_load_ms:.2f}ms | "
            f"synth={t_synth_ms:.2f}ms | total={total_ms:.2f}ms"
        )

        response = Response(wav_bytes, mimetype="audio/wav")
        response.headers["X-Text-Limited"] = str(was_limited).lower()
        response.headers["X-TTS-Synth-Ms"] = f"{t_synth_ms:.1f}"
        return response
    except Exception as e:
        return jsonify({"error": f"Synthesis inference error: {str(e)}"}), 500


@app.get("/api/voice/status")
def voice_status():
    return jsonify({
        "whisper_loaded": _whisper_model is not None,
        "piper_loaded": _piper_voice is not None,
        "device": device,
        "whisper_model": WHISPER_MODEL_NAME,
        "piper_voice": PIPER_VOICE_FILE,
    })


if __name__ == "__main__":
    flask_debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "8000"))

    if flask_debug:
        print("[WARNING] Running in DEBUG mode. Do not expose this server to public tunnels or untrusted networks.")
    else:
        print(f"Starting server in secure standard mode at http://{host}:{port}")

    app.run(host=host, port=port, debug=flask_debug, use_reloader=False)
