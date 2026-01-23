import gradio as gr
import torch
from faster_whisper import WhisperModel
import librosa
import time
import json
import os

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

# -----------------------------
# Prompt handling
# -----------------------------
def build_initial_prompt(odjel, tema):
    prompt_parts = []

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"prompts")

    def load_json(filename):
        if not filename:
            return []

        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Expecting: [{ "text": "..." }, ...]
        return [
            item["text"].strip()
            for item in data
            if isinstance(item, dict) and "text" in item
        ]

    
    if odjel.lower() == "opća":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/intern")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"misc.json"))
        elif tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"misc.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        elif tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/doctor.json"))
            prompt_parts.extend(load_json(f"consults/patient.json"))

    if odjel.lower() == "kardiologija":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/cardio")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))
    
    if odjel.lower() == "pulmologija":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/pulmo")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))

    if odjel.lower() == "gastrologija":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/gastro")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))

    if odjel.lower() == "neurologija":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/neuro")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))

    if odjel.lower() == "infektologija":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/infecto")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))

    if odjel.lower() == "hitna medicina":
        prompt_parts.extend(combine_jsons(os.path.join(base_dir,"medTerm/departments/emergency")))
        if tema.lower() == "opće":
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "bilješke":
            prompt_parts.extend(load_json(f"medTerm/lijekovi/lijekovi.json"))
            prompt_parts.extend(load_json(f"medTerm/departments/general.json"))
        if tema.lower() == "konzultacije":
            prompt_parts.extend(load_json(f"consults/genericMedTerm.json"))
        
    return " ".join(prompt_parts)

def combine_jsons(folder_path):
    combined_texts = []

    if not os.path.exists(folder_path):
        return combined_texts  # Folder does not exist, return empty list

    # Loop over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Extract "text" field
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        combined_texts.append(item["text"].strip())
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

    return combined_texts

# -----------------------------
# Transcription logic
# -----------------------------
def transcribe_file(audio_path, initial_prompt):
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
        initial_prompt=initial_prompt
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

    return transcript, f"Vrijeme za transkripciju: {elapsed:.2f} s"

def transcribe_append(audio_path, prev_transcript, odjel, tema):
    initial_prompt = "Ovo je medicinski transkript na standardnom hrvatskom jeziku" + build_initial_prompt(odjel, tema)

    new_transcript, elapsed_text = transcribe_file(
        audio_path,
        initial_prompt
    )

    combined = (
        prev_transcript + "\n" + new_transcript
        if prev_transcript else new_transcript
    )

    return combined, elapsed_text, combined

def reset_transcript():
    return "", "", ""

# -----------------------------
# GUI
# -----------------------------
with gr.Blocks(
    css=".small-btn { width: 30% !important; }"
) as demo:

    gr.Markdown("## AI MED CRO")

    with gr.Row():
        odjel_dropdown = gr.Dropdown(
            label="Odjel",
            choices=[
                "Opća",
                "Kardiologija",
                "Pulmologija",
                "Gastrologija",
                "Neurologija",
                "Infektologija",
                "Hitna medicina"
            ],
            value="Opća"
        )

        tema_dropdown = gr.Dropdown(
            label="Tema",
            choices=[
                "Opće",
                "Konzultacije",
                "Bilješke"
            ],
            value="Opće"
        )

    audio_input = gr.Audio(
        sources=["microphone", "upload"],
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

    transcribe_button = gr.Button(
        "Zapiši",
        elem_classes="small-btn"
    )

    reset_button = gr.Button(
        "Resetiraj transkript",
        elem_classes="small-btn"
    )

    cumulative_transcript = gr.State("")

    transcribe_button.click(
        fn=transcribe_append,
        inputs=[
            audio_input,
            cumulative_transcript,
            odjel_dropdown,
            tema_dropdown
        ],
        outputs=[
            transcript_output,
            time_output,
            cumulative_transcript
        ]
    )

    reset_button.click(
        fn=reset_transcript,
        inputs=None,
        outputs=[
            transcript_output,
            time_output,
            cumulative_transcript
        ]
    )

demo.launch(theme=gr.themes.Glass())
