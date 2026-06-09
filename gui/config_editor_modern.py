from __future__ import annotations

from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any

try:
    import customtkinter as ctk
except ModuleNotFoundError:
    ctk = None

from gui.config_schema import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    SCHEMA,
    FieldSpec,
    load_yaml,
    update_yaml_scalars,
)


class ModernConfigEditor:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        if ctk is None:
            raise RuntimeError(
                "customtkinter est manquant. Installe les dépendances avec : pip install -r requirements.txt"
            )

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.config_path = config_path
        self.app = ctk.CTk()
        self.app.title("Filtre-Voix-DL - Dataset Studio")
        self.app.geometry("1180x780")
        self.app.minsize(1020, 680)

        self.variables: dict[tuple[str, str], tk.Variable] = {}
        self.field_specs: dict[tuple[str, str], FieldSpec] = {}
        self.value_labels: dict[tuple[str, str], ctk.CTkLabel] = {}
        self.log_queue: queue.Queue[str | None] = queue.Queue()
        self.pipeline_running = False

        self._build_layout()
        self.reload_config(show_message=False)

    def run(self) -> None:
        self.app.mainloop()

    def _build_layout(self) -> None:
        self.app.grid_columnconfigure(1, weight=1)
        self.app.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self.app, width=270, corner_radius=0, fg_color="#111827")
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(8, weight=1)

        ctk.CTkLabel(
            sidebar,
            text="Dataset Studio",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#f9fafb",
        ).grid(row=0, column=0, sticky="w", padx=24, pady=(28, 4))

        ctk.CTkLabel(
            sidebar,
            text="Configure, valide, génère.",
            font=ctk.CTkFont(size=13),
            text_color="#9ca3af",
        ).grid(row=1, column=0, sticky="w", padx=24, pady=(0, 24))

        self.save_button = ctk.CTkButton(
            sidebar,
            text="Sauvegarder",
            height=42,
            command=self.save_config,
        )
        self.save_button.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))

        self.generate_button = ctk.CTkButton(
            sidebar,
            text="Générer dataset final",
            height=42,
            fg_color="#16a34a",
            hover_color="#15803d",
            command=self.run_generated_pipeline,
        )
        self.generate_button.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 10))

        self.full_pipeline_button = ctk.CTkButton(
            sidebar,
            text="Pipeline complet",
            height=42,
            fg_color="#f97316",
            hover_color="#ea580c",
            command=self.run_full_pipeline,
        )
        self.full_pipeline_button.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 10))

        ctk.CTkButton(
            sidebar,
            text="Recharger config",
            height=38,
            fg_color="#374151",
            hover_color="#4b5563",
            command=self.reload_config,
        ).grid(row=5, column=0, sticky="ew", padx=20, pady=(12, 8))

        ctk.CTkButton(
            sidebar,
            text="Copier commande",
            height=38,
            fg_color="#374151",
            hover_color="#4b5563",
            command=self.copy_command,
        ).grid(row=6, column=0, sticky="ew", padx=20, pady=(0, 8))

        self.status_var = tk.StringVar(value="Prêt")
        self.status_label = ctk.CTkLabel(
            sidebar,
            textvariable=self.status_var,
            text_color="#d1d5db",
            anchor="w",
        )
        self.status_label.grid(row=7, column=0, sticky="ew", padx=24, pady=(16, 0))

        ctk.CTkLabel(
            sidebar,
            text="Commande génération rapide :\npython scripts/run_full_pipeline.py --skip-clean --skip-noise --reset-generated",
            text_color="#9ca3af",
            justify="left",
            wraplength=220,
        ).grid(row=9, column=0, sticky="sw", padx=24, pady=24)

        main = ctk.CTkFrame(self.app, fg_color="#f3f4f6", corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(main, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 12))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Configuration du dataset",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#111827",
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text=str(self.config_path),
            font=ctk.CTkFont(size=12),
            text_color="#6b7280",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.tabs = ctk.CTkTabview(main, segmented_button_selected_color="#2563eb")
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 12))

        for tab_name, fields in SCHEMA.items():
            tab = self.tabs.add(tab_name)
            tab.grid_columnconfigure(0, weight=1)
            self._build_tab(tab, tab_name, fields)

        logs = ctk.CTkFrame(main, fg_color="#111827", corner_radius=12)
        logs.grid(row=2, column=0, sticky="nsew", padx=28, pady=(0, 24))
        logs.grid_columnconfigure(0, weight=1)
        logs.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            logs,
            text="Logs pipeline",
            text_color="#f9fafb",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))

        self.log_text = ctk.CTkTextbox(
            logs,
            height=155,
            fg_color="#020617",
            text_color="#e5e7eb",
            border_width=0,
            corner_radius=8,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.log_text.configure(state="disabled")

    def _build_tab(self, parent: ctk.CTkFrame, tab_name: str, fields: list[FieldSpec]) -> None:
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        scroll.grid_columnconfigure((0, 1), weight=1)

        for index, spec in enumerate(fields):
            card = ctk.CTkFrame(scroll, fg_color="#ffffff", corner_radius=10, border_width=1, border_color="#e5e7eb")
            card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=8, pady=8)
            card.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                card,
                text=spec.label,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="#111827",
                anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 2))

            if spec.help_text:
                ctk.CTkLabel(
                    card,
                    text=spec.help_text,
                    text_color="#6b7280",
                    font=ctk.CTkFont(size=12),
                    anchor="w",
                    justify="left",
                    wraplength=360,
                ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))

            key = (spec.section, spec.key)
            self.field_specs[key] = spec
            variable = self._make_variable(spec)
            self.variables[key] = variable

            control_row = 2
            if spec.kind == "bool":
                ctk.CTkSwitch(card, text="", variable=variable).grid(row=control_row, column=0, sticky="w", padx=14, pady=(4, 14))
            elif spec.kind == "prob":
                value_label = ctk.CTkLabel(card, text="0%", text_color="#2563eb", font=ctk.CTkFont(size=13, weight="bold"))
                value_label.grid(row=control_row, column=1, sticky="e", padx=14, pady=(4, 14))
                self.value_labels[key] = value_label

                slider = ctk.CTkSlider(
                    card,
                    from_=0.0,
                    to=1.0,
                    variable=variable,
                    command=lambda value, k=key: self._update_probability_label(k, value),
                )
                slider.grid(row=control_row, column=0, sticky="ew", padx=14, pady=(4, 14))
            else:
                entry = ctk.CTkEntry(card, textvariable=variable, height=36)
                entry.grid(row=control_row, column=0, sticky="ew", padx=14, pady=(4, 14))
                if spec.path_picker:
                    ctk.CTkButton(
                        card,
                        text="Parcourir",
                        width=92,
                        command=lambda v=variable: self.pick_folder(v),
                    ).grid(row=control_row, column=1, sticky="e", padx=(0, 14), pady=(4, 14))

    def _make_variable(self, spec: FieldSpec) -> tk.Variable:
        if spec.kind == "bool":
            return tk.BooleanVar()
        if spec.kind == "prob":
            return tk.DoubleVar()
        return tk.StringVar()

    def _update_probability_label(self, key: tuple[str, str], value: float) -> None:
        if key in self.value_labels:
            self.value_labels[key].configure(text=f"{float(value) * 100:.0f}%")

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
                    numeric_value = float(value)
                    variable.set(numeric_value)
                    self._update_probability_label(key, numeric_value)
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
            updates[key] = (self._coerce_value(spec, variable.get()), spec.kind)

        if updates[("generation", "snr_min_db")][0] > updates[("generation", "snr_max_db")][0]:
            raise ValueError("Le SNR min doit être inférieur ou égal au SNR max.")

        if updates[("augmentations", "compressor_threshold_min_db")][0] > updates[("augmentations", "compressor_threshold_max_db")][0]:
            raise ValueError("Le seuil min de compression doit être inférieur ou égal au seuil max.")

        if updates[("augmentations", "compressor_ratio_min")][0] > updates[("augmentations", "compressor_ratio_max")][0]:
            raise ValueError("Le ratio min de compression doit être inférieur ou égal au ratio max.")

        return updates

    def write_config(self) -> None:
        update_yaml_scalars(self.config_path, self.collect_updates())

    def save_config(self) -> None:
        try:
            self.write_config()
            messagebox.showinfo("Configuration", "Configuration sauvegardée.")
        except Exception as exc:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder :\n{exc}")

    def copy_command(self) -> None:
        command = "python scripts/run_full_pipeline.py --skip-clean --skip-noise --reset-generated"
        self.app.clipboard_clear()
        self.app.clipboard_append(command)
        messagebox.showinfo("Commande", "Commande copiée dans le presse-papiers.")

    def set_running_state(self, running: bool) -> None:
        self.pipeline_running = running
        state = "disabled" if running else "normal"
        self.generate_button.configure(state=state)
        self.full_pipeline_button.configure(state=state)
        self.save_button.configure(state=state)

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def run_generated_pipeline(self) -> None:
        self.start_pipeline(
            [
                sys.executable,
                "scripts/run_full_pipeline.py",
                "--skip-clean",
                "--skip-noise",
                "--reset-generated",
            ],
            "Génération du dataset final",
        )

    def run_full_pipeline(self) -> None:
        if not messagebox.askyesno(
            "Pipeline complet",
            "Le pipeline complet supprime et recrée chunks, metadata, logs et dataset généré. Continuer ?",
        ):
            return

        self.start_pipeline(
            [
                sys.executable,
                "scripts/run_full_pipeline.py",
                "--compile",
                "--reset-all",
            ],
            "Pipeline complet",
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

        thread = threading.Thread(target=self._run_process_worker, args=(command,), daemon=True)
        thread.start()
        self.app.after(100, self.poll_log_queue)

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
            self.app.after(100, self.poll_log_queue)


def main() -> None:
    ModernConfigEditor().run()


if __name__ == "__main__":
    main()
