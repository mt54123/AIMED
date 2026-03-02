import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from datetime import datetime
import os
import threading
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
import librosa
import time

# ---------------- BOJE ----------------
BG_COLOR = "#1e1e1e"
FG_COLOR = "#ffffff"
BTN_GREEN = "#2ecc71"
BTN_RED = "#e74c3c"
BTN_BLUE = "#3498db"
BTN_YELLOW = "#f1c40f"
BTN_HOVER = "#555555"
ENTRY_BG = "#2c2c2c"
ENTRY_FG = "#ffffff"

# ---------------- GLOBAL VARIABLES ----------------
snima = False
pauza = False
snimanje_vrijeme = 0
timer_id = None
RECORDING = {"active": False, "paused": False, "tempfile": None, "data": []}
transcript_text = ""
uploaded_file = None

# ---------------- WHISPER MODEL ----------------
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

# ---------------- HELPER FUNCTIONS ----------------
def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def normalize_text(text):
    return text.replace("  ", " ").strip()

def transcribe_file(audio_path):
    global transcript_text
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
    transcript_text = "\n".join(lines)
    elapsed = time.time() - start_time
    elapsed_text = f"Vrijeme transkripcije: {elapsed:.2f} sekundi"
    return transcript_text, elapsed_text

# ---------------- AUDIO RECORDING ----------------
def record_thread():
    def callback(indata, frames, time_info, status):
        if RECORDING["active"] and not RECORDING["paused"]:
            RECORDING["data"].append(indata.copy())
        elif not RECORDING["active"]:
            raise sd.CallbackStop()
    with sd.InputStream(channels=1, samplerate=TARGET_SR, callback=callback):
        while RECORDING["active"]:
            sd.sleep(50)

def toggle_recording():
    global snima, pauza, RECORDING, snimanje_vrijeme, timer_id
    if not snima:
        # Start recording
        RECORDING["data"] = []
        RECORDING["tempfile"] = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        RECORDING["active"] = True
        RECORDING["paused"] = False
        snima = True
        pauza = False
        snimanje_vrijeme = 0
        update_timer()
        threading.Thread(target=record_thread, daemon=True).start()
        snimi_btn.config(text="PAUSE", bg=BTN_YELLOW)
        status_label.config(text="Snimanje u tijeku...")
    else:
        # Toggle pause
        pauza = not pauza
        if pauza:
            snimi_btn.config(text="RESUME", bg=BTN_BLUE)
            status_label.config(text="Snimanje pauzirano")
        else:
            snimi_btn.config(text="PAUSE", bg=BTN_YELLOW)
            status_label.config(text="Snimanje u tijeku...")

def stop_recording():
    global snima, RECORDING
    if snima:
        RECORDING["active"] = False
        snima = False
        snimi_btn.config(text="SNIMI", bg=BTN_GREEN)
        status_label.config(text="Snimanje završeno")
        if RECORDING["data"]:
            all_data = np.concatenate(RECORDING["data"], axis=0)
            sf.write(RECORDING["tempfile"].name, all_data, TARGET_SR)

def get_audio_file():
    global uploaded_file
    if uploaded_file:
        return uploaded_file
    if RECORDING["tempfile"] and not RECORDING["active"]:
        return RECORDING["tempfile"].name
    return None

def upload_file():
    global uploaded_file
    uploaded_file = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3")])
    if uploaded_file:
        status_label.config(text=f"Učitan file: {os.path.basename(uploaded_file)}")

# ---------------- TIMER ----------------
def update_timer():
    global snimanje_vrijeme, timer_id
    if snima and not pauza:
        minutes = snimanje_vrijeme // 60
        seconds = snimanje_vrijeme % 60
        timer_label.config(text=f"Vrijeme snimanja: {minutes:02d}:{seconds:02d}")
        snimanje_vrijeme += 1
    timer_id = root.after(1000, update_timer)

# ---------------- DOCUMENT ----------------
def generiraj_dokument():
    if not transcript_box.get("1.0", tk.END).strip():
        messagebox.showwarning("Greška", "Prvo transkribiraj audio!")
        return
    dokument_box.delete("1.0", tk.END)
    template = f"""
MEDICINSKI IZVJEŠTAJ
-------------------
Pacijent: {ime_entry.get()} {prezime_entry.get()}
OIB pacijenta: {id_entry.get()}
Liječnik: {lijecnik_entry.get()}
Datum: {datum_label['text']}

TRANSKRIPT:
{transcript_box.get("1.0", tk.END)}

PREPORUKE:
Analgetici i mirovanje.
"""
    dokument_box.insert(tk.END, template)
    status_label.config(text="Dokument generiran")

def spremi_dokument():
    if not dokument_box.get("1.0", tk.END).strip():
        messagebox.showwarning("Greška", "Nema dokumenta za spremiti!")
        return
    ime = ime_entry.get()
    prezime = prezime_entry.get()
    if not ime or not prezime:
        messagebox.showwarning("Greška", "Unesite ime i prezime pacijenta")
        return
    filename = f"{datetime.now().date()}_{ime}_{prezime}_dijagnoza.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(dokument_box.get("1.0", tk.END))
    messagebox.showinfo("Spremljeno", f"Dokument spremljen kao:\n{filename}")

def transkribiraj_manual():
    audio_file = get_audio_file()
    if not audio_file:
        messagebox.showwarning("Greška", "Nema snimljenog ili učitanog audio zapisa")
        return
    transcript, elapsed = transcribe_file(audio_file)
    transcript_box.delete("1.0", tk.END)
    transcript_box.insert(tk.END, transcript)
    status_label.config(text=elapsed)
    # Omogući dokument gumbi nakon transkripta
    gen_btn.config(state="normal")
    save_btn.config(state="normal")

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Medicinski diktat – GUI prototip")
root.geometry("1300x750")
root.configure(bg=BG_COLOR)

# --- Pacijent ---
frame_pacijent = tk.LabelFrame(root, text="Podaci o pacijentu", bg=BG_COLOR, fg=FG_COLOR)
frame_pacijent.pack(fill="x", padx=10, pady=5)

ime_entry = tk.Entry(frame_pacijent, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR)
ime_entry.grid(row=0, column=1, padx=5, pady=2)
tk.Label(frame_pacijent, text="Ime:", bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=0, sticky="e")

prezime_entry = tk.Entry(frame_pacijent, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR)
prezime_entry.grid(row=0, column=3, padx=5, pady=2)
tk.Label(frame_pacijent, text="Prezime:", bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=2, sticky="e")

id_entry = tk.Entry(frame_pacijent, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR)
id_entry.grid(row=1, column=1, padx=5, pady=2)
tk.Label(frame_pacijent, text="OIB:", bg=BG_COLOR, fg=FG_COLOR).grid(row=1, column=0, sticky="e")

lijecnik_entry = tk.Entry(frame_pacijent, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR)
lijecnik_entry.grid(row=1, column=3, padx=5, pady=2)
tk.Label(frame_pacijent, text="Liječnik:", bg=BG_COLOR, fg=FG_COLOR).grid(row=1, column=2, sticky="e")

datum_label = tk.Label(frame_pacijent, text=datetime.now().strftime("%d.%m.%Y"), bg=BG_COLOR, fg=FG_COLOR)
datum_label.grid(row=0, column=4, padx=10)

# --- Diktat gumbi ---
frame_diktat = tk.Frame(root, bg=BG_COLOR)
frame_diktat.pack(fill="x", padx=10, pady=5)

snimi_btn = tk.Button(frame_diktat, text="SNIMI", bg=BTN_GREEN, fg="black", command=toggle_recording)
snimi_btn.pack(side="left", padx=5)

stop_btn = tk.Button(frame_diktat, text="STOP", bg=BTN_RED, fg="black", command=stop_recording)
stop_btn.pack(side="left", padx=5)

upload_btn = tk.Button(frame_diktat, text="UPLOAD FILE", bg=BTN_BLUE, fg="white", command=upload_file)
upload_btn.pack(side="left", padx=5)

transcribe_btn = tk.Button(frame_diktat, text="TRANSKRIBIRAJ", bg=BTN_YELLOW, fg="black", command=transkribiraj_manual)
transcribe_btn.pack(side="left", padx=5)

gen_btn = tk.Button(frame_diktat, text="GENERIRAJ DOKUMENT", bg=BTN_BLUE, fg="white", command=generiraj_dokument, state="disabled")
gen_btn.pack(side="left", padx=5)

save_btn = tk.Button(frame_diktat, text="SPREMI DOKUMENT", bg=BTN_YELLOW, fg="black", command=spremi_dokument, state="disabled")
save_btn.pack(side="left", padx=5)

status_label = tk.Label(frame_diktat, text="Status: spremno", bg=BG_COLOR, fg=FG_COLOR)
status_label.pack(side="left", padx=10)

timer_label = tk.Label(frame_diktat, text="Vrijeme snimanja: 00:00", bg=BG_COLOR, fg=FG_COLOR)
timer_label.pack(side="left", padx=10)

# --- Transkript i dokument (side by side) ---
frame_main = tk.Frame(root, bg=BG_COLOR)
frame_main.pack(fill="both", expand=True, padx=10, pady=5)

# Transkript
frame_transkript = tk.LabelFrame(frame_main, text="Transkript", bg=BG_COLOR, fg=FG_COLOR)
frame_transkript.pack(side="left", fill="both", expand=True, padx=5, pady=5)

transcript_box = tk.Text(frame_transkript, bg=ENTRY_BG, fg=ENTRY_FG, wrap="word")
transcript_box.pack(fill="both", expand=True, side="left", padx=5, pady=5)
scroll_trans = ttk.Scrollbar(frame_transkript, command=transcript_box.yview)
scroll_trans.pack(side="right", fill="y")
transcript_box['yscrollcommand'] = scroll_trans.set

# Dokument
frame_dokument = tk.LabelFrame(frame_main, text="Medicinski dokument", bg=BG_COLOR, fg=FG_COLOR)
frame_dokument.pack(side="left", fill="both", expand=True, padx=5, pady=5)

dokument_box = tk.Text(frame_dokument, bg=ENTRY_BG, fg=ENTRY_FG, wrap="word", height=30)
dokument_box.pack(fill="both", expand=True, side="left", padx=5, pady=5)
scroll_doc = ttk.Scrollbar(frame_dokument, command=dokument_box.yview)
scroll_doc.pack(side="right", fill="y")
dokument_box['yscrollcommand'] = scroll_doc.set

root.mainloop()
