const form = document.getElementById("chat-form");
const input = document.getElementById("message");
const chat = document.getElementById("chat");
const statusEl = document.getElementById("status");
const resetBtn = document.getElementById("reset-btn");
const debugToggle = document.getElementById("debug-toggle");

const dbgStatus = document.getElementById("dbg-status");
const dbgScore = document.getElementById("dbg-score");
const dbgChunks = document.getElementById("dbg-chunks");
const dbgHistory = document.getElementById("dbg-history");
const dbgContext = document.getElementById("dbg-context");
const dbgSession = document.getElementById("dbg-session");
const debugChunksEl = document.getElementById("debug-chunks");
const debugContextEl = document.getElementById("debug-context");

const ringProgress = document.getElementById("ring-progress");
const ringLabel = document.getElementById("ring-label");

// Voice Elements
const voiceToggleBtn = document.getElementById("voice-toggle-btn");
const voiceBtnLabel = document.getElementById("voice-btn-label");
const voiceBar = document.getElementById("voice-bar");
const voiceDot = document.getElementById("voice-dot");
const voiceStateText = document.getElementById("voice-state-text");
const voiceWave = document.getElementById("voice-wave");
const voiceCancelBtn = document.getElementById("voice-cancel-btn");

const sessionId = "default";
const RADIUS = 22;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

ringProgress.style.strokeDasharray = `${CIRCUMFERENCE}`;
ringProgress.style.strokeDashoffset = `${CIRCUMFERENCE}`;

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function addMessage(text, role, sources = []) {
    const wrap = document.createElement("div");
    wrap.className = `message ${role}`;

    const textEl = document.createElement("div");
    textEl.textContent = text;
    wrap.appendChild(textEl);

    if (role === "bot" && sources.length) {
        const seen = new Set();
        const uniqueSources = [];
        for (const s of sources) {
            const title = s.section_title || "Unknown";
            const file = s.source_file || "n/a";
            const key = `${title}::${file}`;
            if (!seen.has(key)) {
                seen.add(key);
                uniqueSources.push(s);
            }
        }

        if (uniqueSources.length > 0) {
            const meta = document.createElement("div");
            meta.className = "sources";
            meta.textContent = "Sources: " + uniqueSources
                .map(s => `${s.section_title || "Unknown"} (${s.source_file || "n/a"})`)
                .join(" • ");
            wrap.appendChild(meta);
        }
    }

    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
}

function setRing(fillRatio) {
    const ratio = Math.max(0, Math.min(fillRatio || 0, 1));
    const offset = CIRCUMFERENCE * (1 - ratio);
    ringProgress.style.strokeDashoffset = `${offset}`;
    ringLabel.textContent = `${Math.round(ratio * 100)}%`;

    if (ratio < 0.5) {
        ringProgress.style.stroke = "#22c55e";
    } else if (ratio < 0.8) {
        ringProgress.style.stroke = "#eab308";
    } else {
        ringProgress.style.stroke = "#ef4444";
    }
}

function renderDebug(data) {
    const meta = data.meta || {};
    const debug = data.debug || {};

    dbgStatus.textContent = "Last response";
    dbgScore.textContent = meta.top_score ?? "-";
    dbgChunks.textContent = meta.retrieved_chunks ?? "-";
    dbgHistory.textContent = meta.history_messages ?? "-";
    dbgContext.textContent = meta.context_chars_used && meta.context_char_limit
        ? `${meta.context_chars_used} / ${meta.context_char_limit}`
        : "-";
    dbgSession.textContent = meta.session_chars && meta.session_char_limit
        ? `${meta.session_chars} / ${meta.session_char_limit}`
        : "-";

    setRing(meta.session_fill_ratio || 0);

    const chunks = debug.retrieved_chunks || [];
    if (!chunks.length) {
        debugChunksEl.innerHTML = `<div class="debug-empty">Debug disabled or no chunks returned.</div>`;
    } else {
        debugChunksEl.innerHTML = chunks.map(chunk => `
      <div class="debug-item">
        <div class="debug-item-top">
          <strong>${escapeHtml(chunk.section_title || "Unknown")}</strong>
          <span>${chunk.score ?? "-"}</span>
        </div>
        <div class="debug-item-meta">
          ${escapeHtml(chunk.source_file || "n/a")}
        </div>
        <div class="debug-item-preview">
          ${escapeHtml(chunk.preview || "")}
        </div>
      </div>
    `).join("");
    }

    debugContextEl.textContent = debug.context_preview || "Debug disabled.";
}

async function askQuestion(message, signal = null) {
    statusEl.textContent = "Thinking...";
    dbgStatus.textContent = "Running";

    try {
        const res = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                session_id: sessionId,
                question: message,
                debug: debugToggle.checked
            }),
            signal: signal || undefined
        });

        const data = await res.json();

        if (!res.ok) {
            addMessage(data.error || "Request failed.", "bot");
            return null;
        } else {
            addMessage(data.answer || "No answer returned.", "bot", data.sources || []);
            renderDebug(data);
            return data;
        }
    } catch (err) {
        if (err.name === "AbortError") {
            return null;
        }
        addMessage("Could not reach backend.", "bot");
        dbgStatus.textContent = "Error";
        return null;
    } finally {
        statusEl.textContent = "Ready";
    }
}

async function resetSession() {
    if (voiceActive) {
        stopVoiceMode();
    }

    await fetch("/reset", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ session_id: sessionId })
    });

    chat.innerHTML = "";
    debugChunksEl.innerHTML = `<div class="debug-empty">No request yet.</div>`;
    debugContextEl.textContent = "No request yet.";
    dbgStatus.textContent = "Reset";
    dbgScore.textContent = "-";
    dbgChunks.textContent = "-";
    dbgHistory.textContent = "-";
    dbgContext.textContent = "-";
    dbgSession.textContent = "-";
    setRing(0);
    statusEl.textContent = "Ready";
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    cleanupAudio();
    if (voiceActive) {
        setVoiceState(VoiceState.LISTENING);
    }

    addMessage(message, "user");
    input.value = "";
    input.disabled = true;

    try {
        await askQuestion(message);
    } finally {
        input.disabled = false;
        input.focus();
    }
});

resetBtn.addEventListener("click", resetSession);

// ==========================================
// --- Voice Assistant State & Audio Logic ---
// ==========================================

const VoiceState = {
    IDLE: "idle",
    LISTENING: "listening",
    TRANSCRIBING: "transcribing",
    THINKING: "thinking",
    SPEAKING: "speaking"
};

let currentVoiceState = VoiceState.IDLE;
let voiceActive = false;
let sessionGeneration = 0;
let currentAbortController = null;

let mediaStream = null;
let audioContext = null;
let processorNode = null;
let muteGain = null;
let currentAudio = null;
let currentAudioUrl = null;

let recordedChunks = [];
let hasSpoken = false;
let silenceStart = null;
const PRE_SPEECH_CHUNKS_LIMIT = 4; // ~350-500ms pre-speech ring buffer
const SPEECH_THRESHOLD = 0.015;
const SILENCE_TIMEOUT_MS = 1200;
const MAX_RECORDING_MS = 30000;
let recordingStartTime = 0;

function cleanupAudio() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.onended = null;
        currentAudio.onerror = null;
        currentAudio = null;
    }
    if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
        currentAudioUrl = null;
    }
}

function cleanTextForSpeech(text) {
    if (!text) return "";
    let cleaned = text.replace(/\[\d+\]\s*(section|subsection|file)\s*=[^\n]+/gi, "");
    cleaned = cleaned.replace(/\[\d+\]/g, "");
    cleaned = cleaned.replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1");
    cleaned = cleaned.replace(/[*#_`>~]/g, "");
    cleaned = cleaned.replace(/\s+([.,!?])/g, "$1");
    cleaned = cleaned.replace(/\s+/g, " ").trim();
    return cleaned;
}

function splitIntoSentenceChunks(text, minChunkLen = 45, maxChunkLen = 160) {
    const cleaned = cleanTextForSpeech(text);
    if (!cleaned) return [];
    if (cleaned.length <= 140) return [cleaned];

    const rawSentences = cleaned
        .split(/(?<=[.!?])\s+/)
        .map(s => s.trim())
        .filter(s => s.length > 0);

    if (rawSentences.length <= 1) return [cleaned];

    const chunks = [];
    let current = "";
    for (const s of rawSentences) {
        if (!current) {
            current = s;
        } else if (current.length < minChunkLen) {
            current += " " + s;
        } else if (current.length + s.length + 1 <= maxChunkLen) {
            current += " " + s;
        } else {
            chunks.push(current);
            current = s;
        }
    }
    if (current) {
        chunks.push(current);
    }
    return chunks;
}

function setVoiceState(newState, customLabel) {
    currentVoiceState = newState;
    voiceDot.className = `voice-dot ${newState}`;
    voiceStateText.textContent = customLabel || (
        newState === VoiceState.LISTENING ? "Listening for English speech..." :
        newState === VoiceState.TRANSCRIBING ? "Transcribing speech..." :
        newState === VoiceState.THINKING ? "Consulting handbook..." :
        newState === VoiceState.SPEAKING ? "Assistant speaking..." : "Idle"
    );

    if (newState === VoiceState.LISTENING || newState === VoiceState.SPEAKING) {
        voiceWave.classList.add("active");
    } else {
        voiceWave.classList.remove("active");
    }
}

function downsampleAndEncodeWav(chunks, sourceSampleRate) {
    let totalLength = 0;
    for (const chunk of chunks) totalLength += chunk.length;
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
    }

    let samples16k;
    if (sourceSampleRate === 16000) {
        samples16k = merged;
    } else {
        const ratio = sourceSampleRate / 16000;
        const newLen = Math.round(merged.length / ratio);
        samples16k = new Float32Array(newLen);
        let outIdx = 0, inIdx = 0;
        while (outIdx < newLen) {
            const nextInIdx = Math.round((outIdx + 1) * ratio);
            let sum = 0, count = 0;
            for (let i = inIdx; i < nextInIdx && i < merged.length; i++) {
                sum += merged[i];
                count++;
            }
            samples16k[outIdx] = count > 0 ? (sum / count) : 0;
            outIdx++;
            inIdx = nextInIdx;
        }
    }

    const buffer = new ArrayBuffer(44 + samples16k.length * 2);
    const view = new DataView(buffer);
    function writeStr(pos, str) {
        for (let i = 0; i < str.length; i++) view.setUint8(pos + i, str.charCodeAt(i));
    }
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + samples16k.length * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true);
    view.setUint32(28, 32000, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, samples16k.length * 2, true);

    let p = 44;
    for (let i = 0; i < samples16k.length; i++, p += 2) {
        const s = Math.max(-1, Math.min(1, samples16k[i]));
        view.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([buffer], { type: "audio/wav" });
}

async function playAudioChunks(chunks, turnGeneration, signal) {
    if (!chunks || chunks.length === 0) {
        if (voiceActive && sessionGeneration === turnGeneration) {
            recordedChunks = [];
            hasSpoken = false;
            silenceStart = null;
            recordingStartTime = Date.now();
            setVoiceState(VoiceState.LISTENING);
        }
        return;
    }

    setVoiceState(VoiceState.SPEAKING);

    async function fetchChunkAudio(text) {
        const res = await fetch("/api/voice/synthesize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
            signal: signal
        });
        if (!res.ok) {
            throw new Error(`TTS synthesis HTTP error: ${res.status}`);
        }
        const blob = await res.blob();
        if (!voiceActive || sessionGeneration !== turnGeneration) {
            return null;
        }
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        return { audio, url };
    }

    const tStartPlayback = performance.now();
    let nextFetchPromise = null;
    let nextIndex = 1;

    try {
        const t0 = performance.now();
        const first = await fetchChunkAudio(chunks[0]);
        if (!voiceActive || sessionGeneration !== turnGeneration || !first) {
            if (first && first.url) URL.revokeObjectURL(first.url);
            return;
        }

        const firstChunkDurationMs = performance.now() - t0;
        console.log(`[TTS] Chunk 1/${chunks.length} ready in ${firstChunkDurationMs.toFixed(1)}ms (${chunks[0].length} chars)`);

        // If there are more chunks, start fetching chunk 2 in background right away!
        if (nextIndex < chunks.length) {
            nextFetchPromise = fetchChunkAudio(chunks[nextIndex++]);
        }

        let currentItem = first;
        let chunkNum = 1;

        while (currentItem) {
            if (!voiceActive || sessionGeneration !== turnGeneration) {
                if (currentItem.url) URL.revokeObjectURL(currentItem.url);
                return;
            }

            cleanupAudio();
            currentAudio = currentItem.audio;
            currentAudioUrl = currentItem.url;

            if (chunkNum === 1) {
                console.log(`[TTS] First audio started playing in ${(performance.now() - tStartPlayback).toFixed(1)}ms after answer received`);
            }

            // Play the current audio chunk to completion
            await new Promise((resolve, reject) => {
                currentAudio.onended = () => resolve();
                currentAudio.onerror = (e) => reject(e);
                currentAudio.play().catch(reject);
            });

            // Finished playing this chunk: clean it up
            cleanupAudio();

            if (!voiceActive || sessionGeneration !== turnGeneration) {
                return;
            }

            // Advance to next chunk if available
            if (nextFetchPromise) {
                chunkNum++;
                currentItem = await nextFetchPromise;
                // Prefetch the subsequent chunk while currentItem is prepared
                if (nextIndex < chunks.length) {
                    nextFetchPromise = fetchChunkAudio(chunks[nextIndex++]);
                } else {
                    nextFetchPromise = null;
                }
            } else {
                currentItem = null;
            }
        }

        // All chunks finished playing: resume microphone listening
        if (voiceActive && sessionGeneration === turnGeneration) {
            recordedChunks = [];
            hasSpoken = false;
            silenceStart = null;
            recordingStartTime = Date.now();
            setVoiceState(VoiceState.LISTENING);
        }

    } catch (err) {
        cleanupAudio();
        if (err.name === "AbortError") {
            return;
        }
        console.error("TTS playback queue error:", err);
        if (voiceActive && sessionGeneration === turnGeneration) {
            recordedChunks = [];
            hasSpoken = false;
            silenceStart = null;
            recordingStartTime = Date.now();
            setVoiceState(VoiceState.LISTENING);
        }
    }
}

async function finishSpeechTurn() {
    if (!voiceActive || currentVoiceState !== VoiceState.LISTENING) return;

    const turnGeneration = sessionGeneration;
    if (currentAbortController) {
        currentAbortController.abort();
    }
    currentAbortController = new AbortController();
    const signal = currentAbortController.signal;

    setVoiceState(VoiceState.TRANSCRIBING);
    const wavBlob = downsampleAndEncodeWav(recordedChunks, audioContext ? audioContext.sampleRate : 16000);
    recordedChunks = [];
    hasSpoken = false;
    silenceStart = null;

    try {
        const formData = new FormData();
        formData.append("audio", wavBlob, "speech.wav");

        const res = await fetch("/api/voice/transcribe", {
            method: "POST",
            body: formData,
            signal: signal
        });
        const data = await res.json();

        // Guard against session termination or new turn
        if (!voiceActive || sessionGeneration !== turnGeneration) return;

        if (!res.ok || data.error || !data.text || data.silence) {
            setVoiceState(VoiceState.LISTENING);
            recordingStartTime = Date.now();
            return;
        }

        const userText = data.text.trim();
        addMessage(userText, "user");

        setVoiceState(VoiceState.THINKING);
        const answerData = await askQuestion(userText, signal);

        // Guard against session termination or new turn
        if (!voiceActive || sessionGeneration !== turnGeneration) return;

        const answerText = answerData && answerData.answer ? answerData.answer : "";
        if (!answerText) {
            setVoiceState(VoiceState.LISTENING);
            recordingStartTime = Date.now();
            return;
        }

        const chunks = splitIntoSentenceChunks(answerText);
        await playAudioChunks(chunks, turnGeneration, signal);

    } catch (err) {
        if (err.name === "AbortError") {
            return; // Request was cleanly cancelled by user action
        }
        console.error("Voice pipeline error:", err);
        if (voiceActive && sessionGeneration === turnGeneration) {
            setVoiceState(VoiceState.LISTENING, "Listening for English speech...");
            recordingStartTime = Date.now();
        }
    }
}

async function startVoiceMode() {
    if (voiceActive) return;

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
    } catch (err) {
        alert("Microphone access was denied or is unavailable: " + err.message);
        return;
    }

    voiceActive = true;
    sessionGeneration++;
    const currentGen = sessionGeneration;

    voiceToggleBtn.classList.add("active");
    voiceToggleBtn.setAttribute("title", "Stop Voice Mode");
    if (voiceBtnLabel) voiceBtnLabel.textContent = "Stop Voice";
    voiceBar.classList.remove("hidden");

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === "suspended") {
        await audioContext.resume();
    }

    const source = audioContext.createMediaStreamSource(mediaStream);
    processorNode = audioContext.createScriptProcessor(4096, 1, 1);

    recordedChunks = [];
    hasSpoken = false;
    silenceStart = null;
    recordingStartTime = Date.now();
    setVoiceState(VoiceState.LISTENING);

    processorNode.onaudioprocess = (e) => {
        if (!voiceActive || sessionGeneration !== currentGen) return;

        // CRITICAL: Ignore microphone processing unless explicitly in LISTENING state
        // Prevents any self-transcription loop during TTS speech playback or thinking!
        if (currentVoiceState !== VoiceState.LISTENING) return;

        const inputData = e.inputBuffer.getChannelData(0);

        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
            sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);

        if (rms > SPEECH_THRESHOLD) {
            hasSpoken = true;
            silenceStart = null;
            recordedChunks.push(new Float32Array(inputData));
        } else if (hasSpoken) {
            // User was speaking and is now paused
            recordedChunks.push(new Float32Array(inputData));
            if (silenceStart === null) {
                silenceStart = Date.now();
            } else if (Date.now() - silenceStart > SILENCE_TIMEOUT_MS) {
                finishSpeechTurn();
            }
        } else {
            // Unbounded silence prevention: keep only small pre-speech ring buffer
            recordedChunks.push(new Float32Array(inputData));
            if (recordedChunks.length > PRE_SPEECH_CHUNKS_LIMIT) {
                recordedChunks.shift();
            }
        }

        // Safety limit: auto-finalize if speaking uninterrupted for max duration
        if (hasSpoken && (Date.now() - recordingStartTime > MAX_RECORDING_MS)) {
            finishSpeechTurn();
        }
    };

    source.connect(processorNode);
    muteGain = audioContext.createGain();
    muteGain.gain.value = 0;
    processorNode.connect(muteGain);
    muteGain.connect(audioContext.destination);
}

function stopVoiceMode() {
    voiceActive = false;
    sessionGeneration++;
    currentVoiceState = VoiceState.IDLE;

    voiceToggleBtn.classList.remove("active");
    voiceToggleBtn.setAttribute("title", "Toggle Voice Mode");
    if (voiceBtnLabel) voiceBtnLabel.textContent = "Voice Mode";
    voiceBar.classList.add("hidden");

    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }

    cleanupAudio();

    if (processorNode) {
        processorNode.disconnect();
        processorNode = null;
    }

    if (muteGain) {
        muteGain.disconnect();
        muteGain = null;
    }

    if (audioContext && audioContext.state !== "closed") {
        audioContext.close().catch(() => {});
        audioContext = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => {
            try {
                track.stop();
            } catch (err) {}
        });
        mediaStream = null;
    }

    recordedChunks = [];
    hasSpoken = false;
    silenceStart = null;
}

voiceToggleBtn.addEventListener("click", () => {
    if (voiceActive) {
        stopVoiceMode();
    } else {
        startVoiceMode();
    }
});

voiceCancelBtn.addEventListener("click", stopVoiceMode);

setRing(0);
debugChunksEl.innerHTML = `<div class="debug-empty">No request yet.</div>`;
