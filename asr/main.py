import gradio as gr
import torch
from faster_whisper import WhisperModel
import librosa
import time

# -----------------------------
# Model loading
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperModel(
    "medium",
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)

TARGET_SR = 16000
SUPPRESS_TOKENS = [50256, 50361]

# -----------------------------
# Helper functions
# -----------------------------
def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

TERM_CORRECTIONS = {
    "naziv tvrtke": "NazivTvrtke",
    "tehnicki izraz": "TehničkiIzraz",
    "medicinski termin": "MedicinskiTermin",
    "Bol u koljenu": "Ispravljeni termin"
}

PROMPT = (
    "Ovo je transkript na hrvatskom jeziku. "
    "Ugovoreni pojmovi uključuju NazivTvrtke, TehničkiIzraz i MedicinskiTermin."
)

def transcribe_file(audio_path):
    """Batch transcription for single file"""
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
        lines.append(f"[{start} → {end}] {text}")

    transcript = "\n".join(lines)
    elapsed = time.time() - start_time
    elapsed_text = f"Vrijeme za transkripciju: {elapsed:.2f} s"

    return transcript, elapsed_text

# -----------------------------
# Function for cumulative transcript
# -----------------------------
def transcribe_append(audio_path, prev_transcript):
    new_transcript, elapsed_text = transcribe_file(audio_path)
    combined = prev_transcript + "\n" + new_transcript if prev_transcript else new_transcript
    return combined, elapsed_text, combined

def reset_transcript():
    return "", "", ""

# -----------------------------
# Main App - GUI
# -----------------------------
with gr.Blocks(
    css="""
    .small-btn { width: 30% !important; }""") as demo:
    gr.Markdown("AI MED CRO")
    audio_input = gr.Audio(
        sources=["microphone","upload"],
        type="filepath",
        label="Snimite govor ili učitajte audiosnimku"
    )

    transcript_output = gr.Textbox(
        label="Transkript",
        lines=12
    )

    time_output = gr.Textbox(
        label="Proteklo vrijeme",
        lines=1
    )

    transcribe_button = gr.Button("Zapiši",
                                  elem_classes="small-btn")
    reset_button = gr.Button("Resetiraj transkript",
                             elem_classes="small-btn")

    cumulative_transcript = gr.State("")

    # -----------------------------
    # Event bindings
    # -----------------------------
    transcribe_button.click(
        fn=transcribe_append,
        inputs=[audio_input, cumulative_transcript],
        outputs=[transcript_output, time_output, cumulative_transcript]
    )

    reset_button.click(
        fn=reset_transcript,
        inputs=None,
        outputs=[transcript_output, time_output, cumulative_transcript]
    )

demo.launch(theme=gr.themes.Glass())
