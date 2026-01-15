import gradio as gr
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import threading
import torch
from faster_whisper import WhisperModel
import librosa
import time
import matplotlib.pyplot as plt
import io

# -----------------------------
# Load Whisper medium Croatian model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel("medium", device=device, compute_type="float16" if device=="cuda" else "int8")

TARGET_SR = 16000
SUPPRESS_TOKENS = [50256, 50361]

TERM_CORRECTIONS = {
    "naziv tvrtke": "NazivTvrtke",
    "tehnicki izraz": "TehničkiIzraz",
    "medicinski termin": "MedicinskiTermin",
    "hipertenzija": "Hipertenzija",
    "diabetes mellitus": "Diabetes mellitus",
    "aspirin": "Aspirin"
}

PROMPT = (
    "Ovo je medicinski transkript na hrvatskom jeziku. "
    "Česti medicinski termini uključuju NazivTvrtke, TehničkiIzraz i MedicinskiTermin."
)

# -----------------------------
# Helper functions
# -----------------------------
def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def normalize_text(text):
    return text.replace("  ", " ").strip()

def transcribe_file(audio_path):
    """Batch transcription of full file"""
    if audio_path is None:
        return "", ""

    samples, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    samples = samples.astype(float)

    start_time = time.time()
    segments, _ = model.transcribe(
        samples,
        language="hr",
        beam_size=5,
        suppress_tokens=SUPPRESS_TOKENS,
        initial_prompt=PROMPT
    )

    lines = []
    for seg in segments:
        start = format_time(seg.start)
        end = format_time(seg.end)
        text = seg.text.strip()
        for wrong, correct in TERM_CORRECTIONS.items():
            text = text.replace(wrong, correct)
        text = normalize_text(text)
        lines.append(f"[{start} → {end}] {text}")

    transcript = "\n".join(lines)
    elapsed = time.time() - start_time
    elapsed_text = f"Vrijeme transkripcije: {elapsed:.2f} sekundi"

    return transcript, elapsed_text

# -----------------------------
# Audio recording
# -----------------------------
RECORDING = {"active": False, "tempfile": None, "data": []}

def toggle_recording():
    if not RECORDING["active"]:
        RECORDING["data"] = []
        RECORDING["tempfile"] = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        RECORDING["active"] = True
        threading.Thread(target=record_thread, daemon=True).start()
        return "Stop"
    else:
        RECORDING["active"] = False
        if len(RECORDING["data"]) > 0:
            all_data = np.concatenate(RECORDING["data"], axis=0)
            sf.write(RECORDING["tempfile"].name, all_data, TARGET_SR)
        return "Snimi"

def record_thread():
    def callback(indata, frames, time_info, status):
        if RECORDING["active"]:
            RECORDING["data"].append(indata.copy())
        else:
            raise sd.CallbackStop()

    with sd.InputStream(channels=1, samplerate=TARGET_SR, callback=callback):
        while RECORDING["active"]:
            sd.sleep(50)

def get_audio_file(upload_file):
    if RECORDING["tempfile"] and not RECORDING["active"]:
        return RECORDING["tempfile"].name
    return upload_file

# -----------------------------
# Waveform image
# -----------------------------
def waveform_image(upload_file):
    if RECORDING["active"] and len(RECORDING["data"]) > 0:
        data = np.concatenate(RECORDING["data"], axis=0).flatten()
    else:
        audio_file = get_audio_file(upload_file)
        if audio_file is None:
            return None
        data, _ = sf.read(audio_file)
        data = data.flatten()
    # Normalize
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))
    # Plot
    fig, ax = plt.subplots(figsize=(6,2))
    ax.plot(data)
    ax.set_title("Valni oblik audio signala")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# -----------------------------
# Gradio interface
# -----------------------------
with gr.Blocks() as demo:

    gr.Markdown("## 🎙️ Batch Hrvatski Medicinski Transkript (Whisper Medium)")

    waveform_img = gr.Image(label="Valni oblik audio signala")

    with gr.Row():
        record_btn = gr.Button("Snimi")
        upload_input = gr.File(label="Učitaj audio", file_types=[".wav", ".mp3"])
        reset_btn = gr.Button("Reset transkript")

    transcript_output = gr.Textbox(
        label="Transkript (s vremenskim oznakama)",
        lines=12
    )

    time_output = gr.Textbox(
        label="Vrijeme transkripcije",
        lines=1
    )

    transcribe_button = gr.Button("Transkribiraj")

    cumulative_transcript = gr.State("")

    # -----------------------------
    # Callbacks
    # -----------------------------
    record_btn.click(fn=toggle_recording, inputs=None, outputs=record_btn)
    waveform_update_btn = gr.Button("Update waveform")
    waveform_update_btn.click(fn=waveform_image, inputs=upload_input, outputs=waveform_img)

    # Manual transcription appending to cumulative
    def transcribe_append(upload_file, prev_transcript):
        audio_file = get_audio_file(upload_file)
        new_transcript, elapsed = transcribe_file(audio_file)
        combined = prev_transcript + "\n" + new_transcript if prev_transcript else new_transcript
        return combined, f"Vrijeme transkripcije: {elapsed} sekundi", combined

    transcribe_button.click(
        fn=transcribe_append,
        inputs=[upload_input, cumulative_transcript],
        outputs=[transcript_output, time_output, cumulative_transcript]
    )

    # Reset transcript
    def reset_transcript():
        return "", "", ""
    reset_btn.click(fn=reset_transcript, inputs=None, outputs=[transcript_output, time_output, cumulative_transcript])

demo.launch()
