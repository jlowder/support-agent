# Voice Input Processing

## Architecture

Voice input flows from browser to backend through the Web Speech API and a pluggable backend endpoint.

## Flow

1. User clicks mic button → browser starts recording
2. Web Speech API transcribes audio to text live
3. User releases button or silence triggers → message sends to `/chat`
4. Backend processes text normally through LangGraph agent

## Frontend (index.html)

### Mic button handler
- `micBtn.addEventListener('click', ...)` toggles recording
- Calls `startRecording()` or `stopRecording()`

### Audio capture
```javascript
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
mediaRecorder = new MediaRecorder(stream);
```

### Speech recognition
```javascript
recognition = new SpeechRecognition();
recognition.lang = 'en-US';
recognition.interimResults = true;
```

### Transcription
- `recognition.onresult` receives interim/final transcripts
- Updates input field as user speaks
- `recognition.onend` auto-submits after silence

### Voice output (TTS)
- `/v1/audio/speech` endpoint (Kokoro API)
- Fallback to `window.speechSynthesis` if Kokoro unavailable

## Backend

### `/api/voice/ingress` endpoint

```python
@app.post("/api/voice/ingress")
async def voice_ingress():
    """Voice ingress stub - pending integration."""
```

Returns: `{"status": "success", "message": "Voice ingress is pluggable and pending integration"}`

### Current state

- Voice input: **Working** (browser-side Web Speech API)
- Voice output: **Working** (Kokoro TTS or native browser TTS)
- Voice ingress: **Stub** (pending integration)

## Data path

```
User → mic → MediaRecorder → SpeechRecognition → text input → /chat → agent
```

No audio data reaches the backend. Only transcribed text is sent.

## Browser requirements

- `navigator.mediaDevices.getUserMedia` for mic access
- `SpeechRecognition` or `webkitSpeechRecognition` for transcription

Chrome, Edge, Safari support. Firefox requires manual enable.