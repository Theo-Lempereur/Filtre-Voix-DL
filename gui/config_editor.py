from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_config.yaml"


@dataclass(frozen=True)
class FieldSpec:
    section: str
    key: str
    label: str
    kind: str
    help_text: str = ""
    min_value: float | None = None
    max_value: float | None = None
    step: float = 1.0
    path_picker: bool = False


SCHEMA: dict[str, list[FieldSpec]] = {
    "Sources": [
        FieldSpec("clean_preprocessing", "input_dir", "Dossier voix clean", "str", "Banque d'audios de voix propres.", path_picker=True),
        FieldSpec("noise_preprocessing", "input_dir", "Dossier bruits", "str", "Banque de bruits de fond.", path_picker=True),
        FieldSpec("clean_preprocessing", "max_files", "Max fichiers clean", "int", "Nombre de fichiers clean utilisés. Vide = tous.", 0, 100000),
        FieldSpec("noise_preprocessing", "max_files", "Max fichiers bruit", "int", "Nombre de fichiers bruit utilisés. Vide = tous.", 0, 100000),
        FieldSpec("clean_preprocessing", "shuffle_files", "Mélanger clean", "bool"),
        FieldSpec("noise_preprocessing", "shuffle_files", "Mélanger bruits", "bool"),
        FieldSpec("clean_preprocessing", "random_seed", "Seed clean", "int"),
        FieldSpec("noise_preprocessing", "random_seed", "Seed bruits", "int"),
    ],
    "Préparation": [
        FieldSpec("clean_preprocessing", "sample_rate", "Sample rate", "int", "Fréquence cible en Hz.", 8000, 48000),
        FieldSpec("clean_preprocessing", "chunk_duration_sec", "Durée chunks clean", "float", "Durée des morceaux de voix en secondes.", 0.5, 30.0),
        FieldSpec("noise_preprocessing", "chunk_duration_sec", "Durée chunks bruit", "float", "Durée des morceaux de bruit en secondes.", 0.5, 30.0),
        FieldSpec("clean_preprocessing", "normalize_rms", "Normaliser RMS clean", "bool"),
        FieldSpec("clean_preprocessing", "target_rms_db", "RMS cible clean (dB)", "float", "Volume moyen visé pour les voix.", -60.0, -5.0),
        FieldSpec("clean_preprocessing", "pad_short_files", "Compléter voix courtes", "bool", "À éviter si tu veux des voix naturelles."),
        FieldSpec("noise_preprocessing", "repeat_short_files", "Répéter bruits courts", "bool"),
        FieldSpec("clean_preprocessing", "min_non_silent_ratio", "Ratio non-silence clean", "float", "Filtre les chunks presque silencieux.", 0.0, 1.0, 0.01),
        FieldSpec("noise_preprocessing", "min_non_silent_ratio", "Ratio non-silence bruit", "float", "Filtre les bruits presque silencieux.", 0.0, 1.0, 0.01),
    ],
    "Dataset": [
        FieldSpec("generation", "num_train_samples", "Samples train", "int", "Nombre de paires train noisy/clean.", 0, 1000000),
        FieldSpec("generation", "num_val_samples", "Samples validation", "int"),
        FieldSpec("generation", "num_test_samples", "Samples test", "int"),
        FieldSpec("generation", "snr_min_db", "SNR min (dB)", "float", "Plus bas = bruit plus fort.", -30.0, 40.0),
        FieldSpec("generation", "snr_max_db", "SNR max (dB)", "float", "Plus haut = bruit plus faible.", -30.0, 60.0),
        FieldSpec("generation", "batch_size", "Batch size génération", "int", "Taille des lots pendant la génération.", 1, 4096),
        FieldSpec("generation", "seed", "Seed génération", "int"),
        FieldSpec("generation", "deterministic", "Génération déterministe", "bool"),
        FieldSpec("generation", "skip_existing", "Ignorer fichiers existants", "bool"),
        FieldSpec("generation", "save_noise", "Sauver bruit ajouté", "bool", "Ajoute un dossier noise/ par split."),
    ],
    "Augmentations": [
        FieldSpec("generation", "apply_clean_augment", "Augmenter la voix clean", "bool", "Compression/EQ/téléphone sur la voix avant mix."),
        FieldSpec("generation", "apply_noise_augment", "Augmenter les bruits", "bool", "À activer seulement si tu veux transformer les bruits eux-mêmes."),
        FieldSpec("generation", "apply_post_noisy_augment", "Augmenter l'audio final", "bool", "Simule codec/micro après le mix noisy."),
        FieldSpec("generation", "post_noisy_augment_probability", "Probabilité post-mix", "prob", "Chance d'appliquer les effets au noisy final.", 0.0, 1.0, 0.01),
        FieldSpec("augmentations", "p_gain", "Gain", "prob"),
        FieldSpec("augmentations", "p_eq", "EQ", "prob"),
        FieldSpec("augmentations", "p_compression", "Compression voix", "prob"),
        FieldSpec("augmentations", "p_saturation", "Saturation micro", "prob"),
        FieldSpec("augmentations", "p_clipping", "Clipping", "prob"),
        FieldSpec("augmentations", "p_phone_filter", "Filtre téléphone", "prob"),
        FieldSpec("augmentations", "p_reverb", "Réverbération", "prob"),
        FieldSpec("augmentations", "p_codec", "Codec MP3/Opus", "prob", "Nécessite FFmpeg pour fonctionner."),
        FieldSpec("augmentations", "p_dropout", "Pertes réseau", "prob"),
        FieldSpec("augmentations", "p_quantization", "Quantification cheap", "prob"),
    ],
    "Compression": [
        FieldSpec("augmentations", "compressor_threshold_min_db", "Seuil min (dB)", "float", "Début de compression le plus sensible.", -60.0, 0.0),
        FieldSpec("augmentations", "compressor_threshold_max_db", "Seuil max (dB)", "float", "Début de compression le moins sensible.", -60.0, 0.0),
        FieldSpec("augmentations", "compressor_ratio_min", "Ratio min", "float", "Compression légère.", 1.0, 20.0),
        FieldSpec("augmentations", "compressor_ratio_max", "Ratio max", "float", "Compression forte.", 1.0, 20.0),
        FieldSpec("augmentations", "saturation_drive_min_db", "Saturation min (dB)", "float", "Drive minimum.", 0.0, 30.0),
        FieldSpec("augmentations", "saturation_drive_max_db", "Saturation max (dB)", "float", "Drive maximum.", 0.0, 30.0),
        FieldSpec("augmentations", "phone_highpass_min_hz", "Téléphone HP min", "float", "Coupe-bas minimum.", 20.0, 2000.0),
        FieldSpec("augmentations", "phone_lowpass_max_hz", "Téléphone LP max", "float", "Coupe-haut maximum.", 1000.0, 7900.0),
    ],
}


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML est manquant. Installe les dépendances avec : pip install -r requirements.txt"
        )

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def yaml_scalar(value: Any, kind: str) -> str:
    if kind == "bool":
        return "true" if bool(value) else "false"
    if value is None or value == "":
        return "null"
    if kind in {"int", "float", "prob"}:
        return str(value)
    text = str(value).replace("\\", "/").replace('"', '\\"')
    return f'"{text}"'


def update_yaml_scalars(path: Path, updates: dict[tuple[str, str], tuple[Any, str]]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    section_pattern = re.compile(r"^([A-Za-z_][\w]*):\s*$")
    key_pattern = re.compile(r"^(\s{2})([A-Za-z_][\w]*):(\s*)(.*)$")

    current_section: str | None = None
    seen: set[tuple[str, str]] = set()
    new_lines: list[str] = []

    for line in lines:
        section_match = section_pattern.match(line)
        if section_match:
            current_section = section_match.group(1)
            new_lines.append(line)
            continue

        key_match = key_pattern.match(line)
        if key_match and current_section:
            indent, key, spacing, tail = key_match.groups()
            update_key = (current_section, key)

            if update_key in updates:
                value, kind = updates[update_key]
                comment = ""
                if " #" in tail:
                    comment = "  #" + tail.split(" #", 1)[1]
                new_lines.append(f"{indent}{key}:{spacing}{yaml_scalar(value, kind)}{comment}")
                seen.add(update_key)
                continue

        new_lines.append(line)

    missing = sorted(set(updates) - seen)
    if missing:
        names = ", ".join(f"{section}.{key}" for section, key in missing)
        raise KeyError(f"Clés introuvables dans le YAML : {names}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


class ScrollFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)


class ConfigEditor(tk.Tk):
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        super().__init__()
        self.config_path = config_path
        self.title("Filtre-Voix-DL - Configuration dataset")
        self.geometry("980x720")
        self.minsize(860, 620)

        self.variables: dict[tuple[str, str], tk.Variable] = {}
        self.field_specs: dict[tuple[str, str], FieldSpec] = {}
        self.log_queue: queue.Queue[str | None] = queue.Queue()
        self.pipeline_running = False

        self._configure_style()
        self._build_layout()
        self.reload_config(show_message=False)

    def _configure_style(self) -> None:
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#f6f7f9")
        self.style.configure("Panel.TFrame", background="#ffffff")
        self.style.configure("TLabel", background="#f6f7f9", foreground="#22252a")
        self.style.configure("Panel.TLabel", background="#ffffff", foreground="#22252a")
        self.style.configure("Hint.TLabel", background="#ffffff", foreground="#657080", font=("Segoe UI", 9))
        self.style.configure("Title.TLabel", background="#f6f7f9", foreground="#15171a", font=("Segoe UI", 18, "bold"))
        self.style.configure("Section.TLabel", background="#ffffff", foreground="#15171a", font=("Segoe UI", 12, "bold"))
        self.style.configure("TButton", padding=(10, 6))
        self.style.configure("Accent.TButton", background="#2563eb", foreground="#ffffff")

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=18)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Configuration du dataset", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text=str(self.config_path),
            foreground="#657080",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        actions = ttk.Frame(header)
        actions.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Button(actions, text="Recharger", command=self.reload_config).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text="Sauvegarder", style="Accent.TButton", command=self.save_config).grid(row=0, column=1)

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=1, column=0, sticky="nsew")

        for tab_name, fields in SCHEMA.items():
            frame = ScrollFrame(self.notebook)
            self.notebook.add(frame, text=tab_name)
            self._build_tab(frame.inner, tab_name, fields)

        footer = ttk.Frame(root)
        footer.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        footer.columnconfigure(0, weight=1)

        self.command_var = tk.StringVar(value="python scripts/run_full_pipeline.py --skip-clean --skip-noise --reset-generated")
        command_entry = ttk.Entry(footer, textvariable=self.command_var, state="readonly")
        command_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(footer, text="Copier commande", command=self.copy_command).grid(row=0, column=1)

        run_bar = ttk.Frame(root)
        run_bar.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        run_bar.columnconfigure(2, weight=1)

        self.generate_button = ttk.Button(
            run_bar,
            text="Générer dataset final",
            style="Accent.TButton",
            command=self.run_generated_pipeline,
        )
        self.generate_button.grid(row=0, column=0, padx=(0, 8))

        self.full_pipeline_button = ttk.Button(
            run_bar,
            text="Pipeline complet",
            command=self.run_full_pipeline,
        )
        self.full_pipeline_button.grid(row=0, column=1, padx=(0, 8))

        self.status_var = tk.StringVar(value="Prêt")
        ttk.Label(run_bar, textvariable=self.status_var).grid(row=0, column=2, sticky="w")

        log_panel = ttk.Frame(root)
        log_panel.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        log_panel.columnconfigure(0, weight=1)
        log_panel.rowconfigure(0, weight=1)
        root.rowconfigure(4, weight=0)

        self.log_text = tk.Text(
            log_panel,
            height=9,
            wrap="word",
            state="disabled",
            bg="#111827",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief="flat",
            padx=10,
            pady=8,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar = ttk.Scrollbar(log_panel, orient="vertical", command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

    def _build_tab(self, parent: ttk.Frame, tab_name: str, fields: list[FieldSpec]) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=18)
        panel.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        parent.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)

        ttk.Label(panel, text=tab_name, style="Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))

        row = 1
        for spec in fields:
            key = (spec.section, spec.key)
            self.field_specs[key] = spec

            ttk.Label(panel, text=spec.label, style="Panel.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 16), pady=8)

            variable = self._make_variable(spec)
            self.variables[key] = variable

            if spec.kind == "bool":
                widget = ttk.Checkbutton(panel, variable=variable)
                widget.grid(row=row, column=1, sticky="w", pady=8)
            elif spec.kind == "prob":
                scale = ttk.Scale(panel, from_=0.0, to=1.0, variable=variable)
                scale.grid(row=row, column=1, sticky="ew", pady=8)
                label = ttk.Label(panel, textvariable=self._probability_label_var(variable), style="Panel.TLabel", width=7)
                label.grid(row=row, column=2, sticky="e", padx=(10, 0))
            else:
                entry = ttk.Entry(panel, textvariable=variable)
                entry.grid(row=row, column=1, sticky="ew", pady=8)
                if spec.path_picker:
                    ttk.Button(panel, text="Parcourir", command=lambda v=variable: self.pick_folder(v)).grid(row=row, column=2, sticky="e", padx=(10, 0))

            if spec.help_text:
                row += 1
                ttk.Label(panel, text=spec.help_text, style="Hint.TLabel").grid(row=row, column=1, columnspan=2, sticky="w", pady=(0, 4))

            row += 1

    def _make_variable(self, spec: FieldSpec) -> tk.Variable:
        if spec.kind == "bool":
            return tk.BooleanVar()
        if spec.kind == "prob":
            return tk.DoubleVar()
        return tk.StringVar()

    def _probability_label_var(self, variable: tk.Variable) -> tk.StringVar:
        label_var = tk.StringVar()

        def update(*_args: object) -> None:
            try:
                label_var.set(f"{float(variable.get()) * 100:.0f}%")
            except (tk.TclError, ValueError):
                label_var.set("--")

        variable.trace_add("write", update)
        update()
        return label_var

    def pick_folder(self, variable: tk.Variable) -> None:
        folder = filedialog.askdirectory(initialdir=str(PROJECT_ROOT))
        if folder:
            variable.set(folder.replace("\\", "/"))

    def reload_config(self, show_message: bool = True) -> None:
        try:
            data = load_yaml(self.config_path)
            for key, variable in self.variables.items():
                section, option = key
                spec = self.field_specs[key]
                value = data.get(section, {}).get(option, "")
                if value is None:
                    value = ""
                if spec.kind == "bool":
                    variable.set(bool(value))
                elif spec.kind == "prob":
                    variable.set(float(value))
                else:
                    variable.set(str(value))
            if show_message:
                messagebox.showinfo("Configuration", "Configuration chargée.")
        except Exception as exc:
            messagebox.showerror("Erreur", f"Impossible de charger la configuration :\n{exc}")

    def _coerce_value(self, spec: FieldSpec, raw: Any) -> Any:
        if spec.kind == "bool":
            return bool(raw)
        if spec.kind == "str":
            return str(raw).strip()
        if spec.kind == "int":
            text = str(raw).strip()
            if text == "":
                return None
            value = int(text)
        elif spec.kind in {"float", "prob"}:
            value = float(raw)
        else:
            value = raw

        if value is not None and spec.min_value is not None and value < spec.min_value:
            raise ValueError(f"{spec.label} doit être >= {spec.min_value}.")
        if value is not None and spec.max_value is not None and value > spec.max_value:
            raise ValueError(f"{spec.label} doit être <= {spec.max_value}.")
        if spec.kind == "prob":
            value = round(float(value), 2)
        return value

    def collect_updates(self) -> dict[tuple[str, str], tuple[Any, str]]:
        updates: dict[tuple[str, str], tuple[Any, str]] = {}
        for key, variable in self.variables.items():
            spec = self.field_specs[key]
            value = self._coerce_value(spec, variable.get())
            updates[key] = (value, spec.kind)

        snr_min = updates[("generation", "snr_min_db")][0]
        snr_max = updates[("generation", "snr_max_db")][0]
        if snr_min > snr_max:
            raise ValueError("Le SNR min doit être inférieur ou égal au SNR max.")

        comp_threshold_min = updates.get(("augmentations", "compressor_threshold_min_db"))
        comp_threshold_max = updates.get(("augmentations", "compressor_threshold_max_db"))
        if comp_threshold_min and comp_threshold_max and comp_threshold_min[0] > comp_threshold_max[0]:
            raise ValueError("Le seuil min de compression doit être inférieur ou égal au seuil max.")

        comp_ratio_min = updates.get(("augmentations", "compressor_ratio_min"))
        comp_ratio_max = updates.get(("augmentations", "compressor_ratio_max"))
        if comp_ratio_min and comp_ratio_max and comp_ratio_min[0] > comp_ratio_max[0]:
            raise ValueError("Le ratio min de compression doit être inférieur ou égal au ratio max.")

        return updates

    def save_config(self) -> None:
        try:
            self.write_config()
            messagebox.showinfo("Configuration", "Configuration sauvegardée.")
        except Exception as exc:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder :\n{exc}")

    def write_config(self) -> None:
        updates = self.collect_updates()
        update_yaml_scalars(self.config_path, updates)

    def copy_command(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.command_var.get())
        messagebox.showinfo("Commande", "Commande copiée dans le presse-papiers.")

    def append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def set_running_state(self, running: bool) -> None:
        self.pipeline_running = running
        state = "disabled" if running else "normal"
        self.generate_button.configure(state=state)
        self.full_pipeline_button.configure(state=state)

    def run_generated_pipeline(self) -> None:
        self.start_pipeline(
            [
                sys.executable,
                "scripts/run_full_pipeline.py",
                "--skip-clean",
                "--skip-noise",
                "--reset-generated",
            ],
            title="Génération du dataset final",
        )

    def run_full_pipeline(self) -> None:
        if not messagebox.askyesno(
            "Pipeline complet",
            "Le pipeline complet peut recréer les chunks clean/noise et prendre plus de temps. Continuer ?",
        ):
            return

        self.start_pipeline(
            [
                sys.executable,
                "scripts/run_full_pipeline.py",
                "--compile",
                "--reset-all",
            ],
            title="Pipeline complet",
        )

    def start_pipeline(self, command: list[str], title: str) -> None:
        if self.pipeline_running:
            return

        try:
            self.write_config()
        except Exception as exc:
            messagebox.showerror("Configuration invalide", f"Corrige les paramètres avant de lancer :\n{exc}")
            return

        self.clear_log()
        self.append_log(f"{title}\n")
        self.append_log("Commande : " + " ".join(command) + "\n\n")
        self.status_var.set("Pipeline en cours...")
        self.set_running_state(True)

        thread = threading.Thread(
            target=self._run_process_worker,
            args=(command,),
            daemon=True,
        )
        thread.start()
        self.after(100, self.poll_log_queue)

    def _run_process_worker(self, command: list[str]) -> None:
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                self.log_queue.put(line)

            return_code = process.wait()
            if return_code == 0:
                self.log_queue.put("\n[OK] Pipeline terminé avec succès.\n")
            else:
                self.log_queue.put(f"\n[ERREUR] Pipeline terminé avec le code {return_code}.\n")
        except Exception as exc:
            self.log_queue.put(f"\n[ERREUR] Impossible de lancer le pipeline : {exc}\n")
        finally:
            self.log_queue.put(None)

    def poll_log_queue(self) -> None:
        finished = False

        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if item is None:
                finished = True
            else:
                self.append_log(item)

        if finished:
            self.set_running_state(False)
            self.status_var.set("Prêt")
        else:
            self.after(100, self.poll_log_queue)


def main() -> None:
    app = ConfigEditor()
    app.mainloop()


if __name__ == "__main__":
    main()
