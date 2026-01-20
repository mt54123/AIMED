import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import os

# ---------------- BOJE I DARK MODE ----------------
BG_COLOR = "#1e1e1e"      # tamna pozadina
FG_COLOR = "#ffffff"       # bijeli tekst
BTN_GREEN = "#2ecc71"
BTN_RED = "#e74c3c"

snima = False  # stanje snimanja

# ---------------- DUMMY FUNKCIJE ----------------

def zavrsi_snimanje():
    global snima
    status_label.config(text="Snimanje završeno")
    transkript_box.delete("1.0", tk.END)
    transkript_box.insert(
        tk.END,
        "Pacijent navodi bol u donjem dijelu leđa koja traje nekoliko dana. "
        "Bol se pojačava pri kretanju. Nema povišenu temperaturu."
    )
    snima = False
    snimi_btn.config(text="SNIMI", bg=BTN_GREEN)

def toggle_snimi():
    global snima
    if not snima:
        # KRENI SNIMATI
        snima = True
        snimi_btn.config(text="STOP", bg=BTN_RED)
        status_label.config(text="Snimanje u tijeku...")
        root.after(1500, zavrsi_snimanje)
    else:
        # ZAUSTAVI SNIMANJE
        snima = False
        snimi_btn.config(text="SNIMI", bg=BTN_GREEN)
        status_label.config(text="Snimanje zaustavljeno")

def generiraj_dokument():
    dokument_box.delete("1.0", tk.END)

    template = f"""
MEDICINSKI IZVJEŠTAJ
-------------------
Pacijent: {ime_entry.get()} {prezime_entry.get()}
OIB pacijenta: {id_entry.get()}
Liječnik: {lijecnik_entry.get()}
Datum: {datum_label['text']}

SUBJEKTIVNE TEGOBE:
Pacijent navodi bol u donjem dijelu leđa.

OBJEKTIVNI NALAZ:
Bez vidljivih neuroloških ispada.

DIJAGNOZA:
Lumbalni bolni sindrom.

TERAPIJA:
Analgetici po potrebi, mirovanje.

PREPORUKE:
Kontrola za 7 dana ukoliko simptomi perzistiraju.
"""
    dokument_box.insert(tk.END, template)
    status_label.config(text="Dokument generiran")

def spremi_dokument():
    ime = ime_entry.get()
    prezime = prezime_entry.get()

    if not ime or not prezime:
        messagebox.showwarning("Greška", "Unesite ime i prezime pacijenta")
        return

    filename = f"{datetime.now().date()}_{ime}_{prezime}_dijagnoza.txt"
    file_path = os.path.abspath(filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(dokument_box.get("1.0", tk.END))

    messagebox.showinfo(
        "Spremljeno",
        f"Dokument je uspješno spremljen na lokaciju:\n\n{file_path}"
    )

# ---------------- GUI ----------------

root = tk.Tk()
root.title("Medicinski diktat – GUI prototip")
root.geometry("700x700")
root.configure(bg=BG_COLOR)

# ---------------- PODACI O PACIJENTU ----------------
frame_pacijent = tk.LabelFrame(root, text="Podaci o pacijentu", bg=BG_COLOR, fg=FG_COLOR)
frame_pacijent.pack(fill="x", padx=10, pady=5)

# Red 0: Ime i Prezime
tk.Label(frame_pacijent, text="Ime:", bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=0, padx=5, pady=2)
ime_entry = tk.Entry(frame_pacijent, width=20)
ime_entry.grid(row=0, column=1)

tk.Label(frame_pacijent, text="Prezime:", bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=2, padx=5, pady=2)
prezime_entry = tk.Entry(frame_pacijent, width=20)
prezime_entry.grid(row=0, column=3)

# Red 1: OIB i Liječnik
tk.Label(frame_pacijent, text="OIB pacijenta:", bg=BG_COLOR, fg=FG_COLOR).grid(row=1, column=0, padx=5, pady=2)
id_entry = tk.Entry(frame_pacijent, width=20)
id_entry.grid(row=1, column=1)

tk.Label(frame_pacijent, text="Liječnik:", bg=BG_COLOR, fg=FG_COLOR).grid(row=1, column=2, padx=5, pady=2)
lijecnik_entry = tk.Entry(frame_pacijent, width=20)
lijecnik_entry.grid(row=1, column=3)

# Datum
datum_label = tk.Label(frame_pacijent, text=datetime.now().strftime("%d.%m.%Y"), bg=BG_COLOR, fg=FG_COLOR)
datum_label.grid(row=0, column=4, padx=10)

# ---------------- DIKTAT ----------------
frame_diktat = tk.LabelFrame(root, text="Diktat", bg=BG_COLOR, fg=FG_COLOR)
frame_diktat.pack(fill="x", padx=10, pady=5)

snimi_btn = tk.Button(
    frame_diktat,
    text="SNIMI",
    bg=BTN_GREEN,
    fg="black",
    width=15,
    height=2,
    command=toggle_snimi
)
snimi_btn.pack(side="left", padx=10)

tk.Button(frame_diktat, text="GENERIRAJ DOKUMENT", bg="#3498db", fg="white",
          width=20, height=2, command=generiraj_dokument).pack(side="left", padx=10)

status_label = tk.Label(frame_diktat, text="Status: spremno", bg=BG_COLOR, fg=FG_COLOR)
status_label.pack(side="left", padx=20)

# ---------------- TRANSKRIPT ----------------
frame_transkript = tk.LabelFrame(root, text="Sirovi transkript", bg=BG_COLOR, fg=FG_COLOR)
frame_transkript.pack(fill="both", expand=True, padx=10, pady=5)

transkript_box = tk.Text(frame_transkript, height=6, bg="#2c2c2c", fg="white")
transkript_box.pack(fill="both", expand=True)

# ---------------- DOKUMENT ----------------
frame_dokument = tk.LabelFrame(root, text="Medicinski dokument", bg=BG_COLOR, fg=FG_COLOR)
frame_dokument.pack(fill="both", expand=True, padx=10, pady=5)

dokument_box = tk.Text(frame_dokument, bg="#2c2c2c", fg="white")
dokument_box.pack(fill="both", expand=True)

# ---------------- AKCIJE ----------------
frame_akcije = tk.Frame(root, bg=BG_COLOR)
frame_akcije.pack(pady=10)

tk.Button(frame_akcije, text="SPREMI", bg="#f1c40f", fg="black",
          width=15, height=2, command=spremi_dokument).pack(side="left", padx=10)

tk.Label(
    root,
    text="Sustav služi za ubrzavanje dokumentacije. Liječnik je odgovoran za sadržaj.",
    fg="gray",
    bg=BG_COLOR
).pack(pady=5)

root.mainloop()
