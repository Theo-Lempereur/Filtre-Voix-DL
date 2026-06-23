"""Modern CustomTkinter interface for editing and running the dataset pipeline.

The GUI provides two separated workflows:

* novice mode, where users only choose sample counts per high-level template;
* expert mode, where users edit curated YAML fields with validation.

Pipeline subprocesses are launched from the GUI after saving the current
configuration, and their logs are streamed to a detached log window so the main
form remains readable.
"""

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
    PRESETS,
    PROJECT_ROOT,
    SCHEMA,
    FieldSpec,
    build_template_mix,
    load_yaml,
    update_yaml_template_mix,
    update_yaml_scalars,
)

APP_BG = "#dfe8ec"
SURFACE = "#f8fafc"
CARD = "#ffffff"
TEXT = "#1f2937"
MUTED = "#7b8794"
LINE = "#cfdae2"
BLUE = "#2f80ff"
VIOLET = "#3d2bff"
BLUE_DARK = "#2563eb"
SOFT_BLUE = "#eef5ff"
TAB_DARK = "#334155"
TAB_DARK_HOVER = "#475569"


class LogWindow:
    """Separate pipeline log window kept out of the main configuration flow.

    Attributes:
        parent: Owning ``ModernConfigEditor`` instance.
        window: Lazily created top-level window.
        textbox: Lazily created read-only textbox receiving subprocess output.
    """

    def __init__(self, parent: "ModernConfigEditor") -> None:
        """Store parent references and lazily create widgets when logs are opened.

        Args:
            parent: Main editor that owns the CustomTkinter root window.
        """
        self.parent = parent
        self.window: ctk.CTkToplevel | None = None
        self.textbox: ctk.CTkTextbox | None = None

    def open(self, title: str = "Logs pipeline") -> None:
        """Create or focus the detached log window used by pipeline subprocesses.

        Args:
            title: Window title shown for the current pipeline run.

        Returns:
            None.
        """
        if self.window is not None and self.window.winfo_exists():
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()
            self.window.attributes("-topmost", True)
            self.window.after(250, lambda: self.window.attributes("-topmost", False))
            return

        self.window = ctk.CTkToplevel(self.parent.app)
        self.window.title(title)
        self.window.geometry("920x520")
        self.window.minsize(720, 420)
        self.window.configure(fg_color=APP_BG)
        self.window.transient(self.parent.app)
        self.window.lift()
        self.window.focus_force()
        self.window.attributes("-topmost", True)
        self.window.after(250, lambda: self.window.attributes("-topmost", False))
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.window, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(22, 10))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Logs pipeline",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            header,
            text="Clear",
            width=90,
            fg_color=SURFACE,
            hover_color="#edf2f7",
            text_color=TEXT,
            border_width=1,
            border_color=LINE,
            command=self.clear,
        ).grid(row=0, column=1, sticky="e")

        panel = ctk.CTkFrame(self.window, fg_color=CARD, corner_radius=28, border_width=1, border_color=LINE)
        panel.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 24))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(0, weight=1)

        self.textbox = ctk.CTkTextbox(
            panel,
            fg_color="#0f172a",
            text_color="#e5e7eb",
            border_width=0,
            corner_radius=18,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.textbox.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        self.textbox.configure(state="disabled")

    def clear(self) -> None:
        """Remove all text from the log textbox when the window exists.

        Returns:
            None.
        """
        if self.textbox is None:
            return
        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        self.textbox.configure(state="disabled")

    def append(self, text: str) -> None:
        """Append text to the log window and keep the latest output visible.

        Args:
            text: New log text chunk.

        Returns:
            None.
        """
        self.open()
        if self.textbox is None:
            return
        self.textbox.configure(state="normal")
        self.textbox.insert("end", text)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")


class ModernConfigEditor:
    """Two-level dataset configuration GUI for novice and expert workflows.

    Attributes:
        config_path: YAML file edited by the interface.
        variables: Tk variables keyed by ``(section, option)``.
        field_specs: Schema entries keyed like ``variables``.
        novice_template_vars: Tk variables storing novice template counts.
        log_queue: Thread-safe queue receiving subprocess output.
        pipeline_running: Whether a pipeline subprocess is currently active.
    """

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Create GUI state, widgets, and load the active dataset configuration.

        Args:
            config_path: YAML configuration file edited by the GUI.

        Raises:
            RuntimeError: If ``customtkinter`` is not installed.
        """
        if ctk is None:
            raise RuntimeError(
                "customtkinter is missing. Install dependencies with: pip install -r requirements.txt"
            )

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.config_path = config_path
        self.app = ctk.CTk()
        self.app.title("Wallace - Dataset Studio")
        self.app.geometry("1180x780")
        self.app.minsize(1020, 680)
        self.app.configure(fg_color=APP_BG)

        self.variables: dict[tuple[str, str], tk.Variable] = {}
        self.field_specs: dict[tuple[str, str], FieldSpec] = {}
        self.value_labels: dict[tuple[str, str], ctk.CTkLabel] = {}
        self.novice_template_vars: dict[str, tk.StringVar] = {}
        self.novice_total_label_var = tk.StringVar(value="Total: 0 samples")
        self.log_window = LogWindow(self)
        self.log_queue: queue.Queue[str | None] = queue.Queue()
        self.pipeline_running = False

        self._build_layout()
        self.reload_config(show_message=False)

    def run(self) -> None:
        """Start the CustomTkinter event loop.

        Returns:
            None.
        """
        self.app.mainloop()

    def _build_layout(self) -> None:
        """Build the complete novice/expert configuration interface.

        Returns:
            None.
        """
        self.app.grid_columnconfigure(1, weight=1)
        self.app.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self.app, width=260, corner_radius=34, fg_color=SURFACE, border_width=1, border_color=LINE)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(26, 10), pady=26)
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(12, weight=1)

        ctk.CTkLabel(
            sidebar,
            text="Dataset Studio",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w", padx=24, pady=(28, 4))

        ctk.CTkLabel(
            sidebar,
            text="Configure. Validate. Generate.",
            font=ctk.CTkFont(size=13),
            text_color=MUTED,
        ).grid(row=1, column=0, sticky="w", padx=24, pady=(0, 24))

        self.save_button = ctk.CTkButton(
            sidebar,
            text="Save config",
            height=42,
            fg_color=BLUE,
            hover_color=BLUE_DARK,
            corner_radius=20,
            command=self.save_config,
        )
        self.save_button.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))

        self.generate_button = ctk.CTkButton(
            sidebar,
            text="Generate dataset",
            height=42,
            fg_color=VIOLET,
            hover_color="#3120d8",
            corner_radius=20,
            command=self.run_generated_pipeline,
        )
        self.generate_button.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 10))

        self.full_pipeline_button = ctk.CTkButton(
            sidebar,
            text="Full pipeline",
            height=42,
            fg_color="#ffffff",
            hover_color="#edf2f7",
            text_color=TEXT,
            border_width=1,
            border_color=LINE,
            corner_radius=20,
            command=self.run_full_pipeline,
        )
        self.full_pipeline_button.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 10))

        ctk.CTkButton(
            sidebar,
            text="Reload config",
            height=38,
            fg_color="#ffffff",
            hover_color="#edf2f7",
            text_color=TEXT,
            border_width=1,
            border_color=LINE,
            corner_radius=19,
            command=self.reload_config,
        ).grid(row=5, column=0, sticky="ew", padx=20, pady=(12, 8))

        ctk.CTkButton(
            sidebar,
            text="Copy command",
            height=38,
            fg_color="#ffffff",
            hover_color="#edf2f7",
            text_color=TEXT,
            border_width=1,
            border_color=LINE,
            corner_radius=19,
            command=self.copy_command,
        ).grid(row=6, column=0, sticky="ew", padx=20, pady=(0, 8))

        ctk.CTkButton(
            sidebar,
            text="Open logs",
            height=38,
            fg_color="#ffffff",
            hover_color="#edf2f7",
            text_color=TEXT,
            border_width=1,
            border_color=LINE,
            corner_radius=19,
            command=lambda: self.log_window.open(),
        ).grid(row=7, column=0, sticky="ew", padx=20, pady=(0, 8))

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(
            sidebar,
            textvariable=self.status_var,
            text_color=MUTED,
            anchor="w",
        )
        self.status_label.grid(row=8, column=0, sticky="ew", padx=24, pady=(16, 0))

        ctk.CTkLabel(
            sidebar,
            text="Quick generation:\nrun_full_pipeline.py --skip-clean --skip-noise --reset-generated",
            text_color=MUTED,
            justify="left",
            wraplength=220,
        ).grid(row=13, column=0, sticky="sw", padx=24, pady=24)

        main = ctk.CTkFrame(self.app, fg_color="transparent", corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew", padx=(8, 26), pady=26)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(main, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 12))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Dataset configuration",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text=str(self.config_path),
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.mode_tabs = ctk.CTkTabview(
            main,
            fg_color=SURFACE,
            segmented_button_fg_color=TAB_DARK,
            segmented_button_selected_color=BLUE,
            segmented_button_selected_hover_color=BLUE_DARK,
            segmented_button_unselected_color=TAB_DARK,
            segmented_button_unselected_hover_color=TAB_DARK_HOVER,
            text_color="#ffffff",
            text_color_disabled="#ffffff",
            corner_radius=28,
        )
        self.mode_tabs.grid(row=1, column=0, sticky="nsew", pady=(0, 0))

        novice_tab = self.mode_tabs.add("Novice")
        novice_tab.grid_columnconfigure(0, weight=1)
        novice_tab.grid_rowconfigure(0, weight=1)
        self._build_novice_tab(novice_tab)

        expert_tab = self.mode_tabs.add("Expert")
        expert_tab.grid_columnconfigure(0, weight=1)
        expert_tab.grid_rowconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(
            expert_tab,
            fg_color=SURFACE,
            segmented_button_fg_color=TAB_DARK,
            segmented_button_selected_color=BLUE,
            segmented_button_selected_hover_color=BLUE_DARK,
            segmented_button_unselected_color=TAB_DARK,
            segmented_button_unselected_hover_color=TAB_DARK_HOVER,
            text_color="#ffffff",
            text_color_disabled="#ffffff",
            corner_radius=24,
        )
        self.tabs.grid(row=0, column=0, sticky="nsew")
        for tab_name, fields in SCHEMA.items():
            tab = self.tabs.add(tab_name)
            tab.grid_columnconfigure(0, weight=1)
            self._build_tab(tab, tab_name, fields)

    def _build_novice_tab(self, parent: ctk.CTkFrame) -> None:
        """Build the simplified workflow around three dataset styles.

        Args:
            parent: Tab frame that receives the novice controls.

        Returns:
            None. Widgets and Tk variables are stored on the editor instance.
        """

        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        scroll.grid_columnconfigure(0, weight=1)

        intro = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=28, border_width=1, border_color=LINE)
        intro.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 12))
        intro.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            intro,
            text="Novice mode",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=18, pady=(16, 4))

        ctk.CTkLabel(
            intro,
            text="Enter how many examples you want for each style. Totals and train / validation / test splits are computed automatically.",
            text_color=MUTED,
            justify="left",
            wraplength=760,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=18, pady=(0, 14))

        total_badge = ctk.CTkFrame(intro, fg_color=SOFT_BLUE, corner_radius=18)
        total_badge.grid(row=2, column=0, columnspan=3, sticky="ew", padx=18, pady=(0, 16))
        ctk.CTkLabel(
            total_badge,
            textvariable=self.novice_total_label_var,
            text_color=BLUE_DARK,
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=10)

        grid = ctk.CTkFrame(scroll, fg_color="transparent")
        grid.grid(row=1, column=0, sticky="ew")
        grid.grid_columnconfigure((0, 1, 2), weight=1)

        defaults = {
            "Classic": 60,
            "Microphone mode": 25,
            "Very noisy": 15,
        }

        descriptions = {
            "Classic": "Clean speech mixed with regular background noise.",
            "Microphone mode": "Speech degraded like a cheap mic, phone, or compressed call.",
            "Very noisy": "Hard samples where noise covers more of the voice.",
        }

        colors = {
            "Classic": ("#eefbf5", "#10a66a"),
            "Microphone mode": ("#eef5ff", BLUE),
            "Very noisy": ("#f2efff", VIOLET),
        }

        for index, preset_name in enumerate(PRESETS):
            var = tk.StringVar(value=str(defaults.get(preset_name, 0)))
            var.trace_add("write", lambda *_args: self.update_novice_total())
            self.novice_template_vars[preset_name] = var

            bg_color, accent_color = colors.get(preset_name, ("#ffffff", "#2563eb"))
            card = ctk.CTkFrame(grid, fg_color=CARD, corner_radius=30, border_width=1, border_color=LINE)
            card.grid(row=0, column=index, sticky="nsew", padx=8, pady=8)
            card.grid_columnconfigure(0, weight=1)

            accent = ctk.CTkFrame(card, height=7, fg_color=accent_color, corner_radius=30)
            accent.grid(row=0, column=0, columnspan=3, sticky="ew")

            badge = ctk.CTkFrame(card, fg_color=bg_color, corner_radius=18)
            badge.grid(row=1, column=0, columnspan=3, sticky="ew", padx=14, pady=(16, 8))
            ctk.CTkLabel(
                badge,
                text=preset_name,
                font=ctk.CTkFont(size=17, weight="bold"),
                text_color=accent_color,
            ).grid(row=0, column=0, sticky="w", padx=12, pady=10)

            ctk.CTkLabel(
                card,
                text=descriptions.get(preset_name, ""),
                text_color=MUTED,
                justify="left",
                wraplength=260,
            ).grid(row=2, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 18))

            ctk.CTkLabel(card, text="Quantity", text_color=TEXT, font=ctk.CTkFont(weight="bold")).grid(row=3, column=0, sticky="w", padx=16, pady=(0, 8))
            ctk.CTkEntry(
                card,
                textvariable=var,
                width=130,
                height=44,
                corner_radius=18,
                border_color=LINE,
                fg_color="#f7fafc",
                font=ctk.CTkFont(size=16, weight="bold"),
            ).grid(row=4, column=0, sticky="w", padx=16, pady=(0, 20))

        actions = ctk.CTkFrame(scroll, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=8, pady=(12, 8))
        actions.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            actions,
            text="Save novice config",
            height=46,
            width=240,
            corner_radius=22,
            fg_color=BLUE,
            hover_color=BLUE_DARK,
            command=self.save_novice_config,
        ).grid(row=0, column=0, sticky="e")

        self.update_novice_total()

    def _build_tab(self, parent: ctk.CTkFrame, tab_name: str, fields: list[FieldSpec]) -> None:
        """Build one expert tab from declarative field specifications.

        Args:
            parent: Tab frame that receives the generated controls.
            tab_name: Human-readable expert tab name.
            fields: Field specifications to render in the tab.

        Returns:
            None. Created variables and field specs are registered on the editor
            instance for later load/save operations.
        """

        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        scroll.grid_columnconfigure((0, 1), weight=1)

        for index, spec in enumerate(fields):
            card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=24, border_width=1, border_color=LINE)
            card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=8, pady=8)
            card.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                card,
                text=spec.label,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=TEXT,
                anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 2))

            if spec.help_text:
                ctk.CTkLabel(
                    card,
                    text=spec.help_text,
                    text_color=MUTED,
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
                value_label = ctk.CTkLabel(card, text="0%", text_color=BLUE, font=ctk.CTkFont(size=13, weight="bold"))
                value_label.grid(row=control_row, column=1, sticky="e", padx=14, pady=(4, 14))
                self.value_labels[key] = value_label

                slider = ctk.CTkSlider(
                    card,
                    from_=0.0,
                    to=1.0,
                    variable=variable,
                    button_color=BLUE,
                    button_hover_color=VIOLET,
                    progress_color=BLUE,
                    fg_color="#d9e3ea",
                    command=lambda value, k=key: self._update_probability_label(k, value),
                )
                slider.grid(row=control_row, column=0, sticky="ew", padx=14, pady=(4, 14))
            else:
                entry = ctk.CTkEntry(card, textvariable=variable, height=38, corner_radius=18, border_color=LINE, fg_color="#f7fafc")
                entry.grid(row=control_row, column=0, sticky="ew", padx=14, pady=(4, 14))
                if spec.path_picker:
                    ctk.CTkButton(
                        card,
                        text="Browse",
                        width=92,
                        fg_color=BLUE,
                        hover_color=BLUE_DARK,
                        corner_radius=18,
                        command=lambda v=variable: self.pick_folder(v),
                    ).grid(row=control_row, column=1, sticky="e", padx=(0, 14), pady=(4, 14))

    def _make_variable(self, spec: FieldSpec) -> tk.Variable:
        """Create a Tk variable type compatible with a schema field.

        Args:
            spec: Field schema entry.

        Returns:
            ``BooleanVar`` for booleans, ``DoubleVar`` for probabilities, and
            ``StringVar`` for all other values.
        """
        if spec.kind == "bool":
            return tk.BooleanVar()
        if spec.kind == "prob":
            return tk.DoubleVar()
        return tk.StringVar()

    def _update_probability_label(self, key: tuple[str, str], value: float) -> None:
        """Synchronize slider percentage labels with their current probability value.

        Args:
            key: ``(section, option)`` field key.
            value: Probability value between 0 and 1.

        Returns:
            None.
        """
        if key in self.value_labels:
            self.value_labels[key].configure(text=f"{float(value) * 100:.0f}%")

    def pick_folder(self, variable: tk.Variable) -> None:
        """Open a folder picker and write the selected path into a Tk variable.

        Args:
            variable: Tk variable receiving the selected folder path.

        Returns:
            None.
        """
        folder = filedialog.askdirectory(initialdir=str(PROJECT_ROOT))
        if folder:
            variable.set(folder.replace("\\", "/"))

    def reload_config(self, show_message: bool = True) -> None:
        """Reload YAML values into expert controls.

        Args:
            show_message: Whether to show a success dialog after loading.

        Returns:
            None. Errors are reported through a message box.
        """
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
                messagebox.showinfo("Configuration", "Configuration loaded.")
        except Exception as exc:
            messagebox.showerror("Error", f"Unable to load configuration:\n{exc}")

    def update_novice_total(self) -> None:
        """Recompute the novice total and derived train/validation/test split.

        Returns:
            None. The displayed total label is updated in place.
        """
        total = 0
        invalid = False

        for variable in self.novice_template_vars.values():
            try:
                value = int(variable.get().strip() or "0")
            except ValueError:
                invalid = True
                value = 0
            total += max(0, value)

        if invalid:
            self.novice_total_label_var.set("Total: invalid value")
        else:
            train_count = int(round(total * 0.80))
            val_count = int(round(total * 0.10))
            test_count = total - train_count - val_count
            self.novice_total_label_var.set(
                f"Total: {total} samples  |  train {train_count} · val {val_count} · test {test_count}"
            )

    def collect_novice_counts(self) -> tuple[int, dict[str, int]]:
        """Read and validate novice template quantities from the GUI.

        Returns:
            Tuple ``(total_count, counts_by_template)``.

        Raises:
            ValueError: If a template count is negative or if all counts are zero.
        """
        counts: dict[str, int] = {}
        for name, variable in self.novice_template_vars.items():
            text = variable.get().strip()
            count = int(text or "0")
            if count < 0:
                raise ValueError(f"The {name} template quantity cannot be negative.")
            counts[name] = count

        count_sum = sum(counts.values())
        if count_sum <= 0:
            raise ValueError("Add at least one sample to a template.")

        return count_sum, counts

    def save_novice_config(self) -> None:
        """Persist novice template counts and derived split sizes into YAML.

        Returns:
            None. Errors are reported through a message box.
        """
        try:
            total, counts = self.collect_novice_counts()
            train_count = int(round(total * 0.80))
            val_count = int(round(total * 0.10))
            test_count = total - train_count - val_count

            updates = {
                ("generation", "num_train_samples"): (train_count, "int"),
                ("generation", "num_val_samples"): (val_count, "int"),
                ("generation", "num_test_samples"): (test_count, "int"),
            }
            update_yaml_scalars(self.config_path, updates)
            update_yaml_template_mix(self.config_path, build_template_mix(counts))
            self.reload_config(show_message=False)
            self.status_var.set("Novice config saved")
            messagebox.showinfo(
                "Novice configuration",
                f"Configuration saved: {train_count} train, {val_count} validation, {test_count} test.",
            )
        except Exception as exc:
            messagebox.showerror("Error", f"Unable to save novice configuration:\n{exc}")

    def _coerce_value(self, spec: FieldSpec, raw: Any) -> Any:
        """Convert raw GUI values to the typed values expected by YAML.

        Args:
            spec: Field schema entry describing type and bounds.
            raw: Raw Tk variable value.

        Returns:
            Typed Python value ready for YAML serialization.

        Raises:
            ValueError: If numeric conversion fails or bounds are violated.
        """
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
            raise ValueError(f"{spec.label} must be >= {spec.min_value}.")
        if value is not None and spec.max_value is not None and value > spec.max_value:
            raise ValueError(f"{spec.label} must be <= {spec.max_value}.")
        if spec.kind == "prob":
            value = round(float(value), 2)
        return value

    def collect_updates(self) -> dict[tuple[str, str], tuple[Any, str]]:
        """Collect expert field updates and validate cross-field constraints.

        Returns:
            Mapping ``(section, key) -> (value, kind)`` used by the YAML patcher.

        Raises:
            ValueError: If related ranges are invalid, such as min SNR greater
            than max SNR.
        """
        updates: dict[tuple[str, str], tuple[Any, str]] = {}
        for key, variable in self.variables.items():
            spec = self.field_specs[key]
            updates[key] = (self._coerce_value(spec, variable.get()), spec.kind)

        chunk_duration = updates.get(("clean_preprocessing", "chunk_duration_sec"))
        if chunk_duration is not None:
            updates[("noise_preprocessing", "chunk_duration_sec")] = chunk_duration
            updates[("generation", "duration_sec")] = chunk_duration

        if updates[("generation", "snr_min_db")][0] > updates[("generation", "snr_max_db")][0]:
            raise ValueError("Min SNR must be lower than or equal to max SNR.")

        if updates[("augmentations", "compressor_threshold_min_db")][0] > updates[("augmentations", "compressor_threshold_max_db")][0]:
            raise ValueError("Min compression threshold must be lower than or equal to max threshold.")

        if updates[("augmentations", "compressor_ratio_min")][0] > updates[("augmentations", "compressor_ratio_max")][0]:
            raise ValueError("Min compression ratio must be lower than or equal to max ratio.")

        return updates

    def write_config(self) -> None:
        """Write expert field values to the YAML configuration file.

        Returns:
            None.

        Raises:
            ValueError: Propagated from validation in ``collect_updates``.
            KeyError: Propagated if an expected YAML key is missing.
        """
        update_yaml_scalars(self.config_path, self.collect_updates())

    def save_config(self) -> None:
        """Save expert configuration changes and report the result to the user.

        Returns:
            None. Errors are reported through a message box.
        """
        try:
            self.write_config()
            messagebox.showinfo("Configuration", "Configuration saved.")
        except Exception as exc:
            messagebox.showerror("Error", f"Unable to save:\n{exc}")

    def copy_command(self) -> None:
        """Copy the recommended final-generation command to the clipboard.

        Returns:
            None.
        """
        command = "python scripts/run_full_pipeline.py --skip-clean --skip-noise --reset-generated"
        self.app.clipboard_clear()
        self.app.clipboard_append(command)
        messagebox.showinfo("Command", "Command copied to clipboard.")

    def set_running_state(self, running: bool) -> None:
        """Enable or disable action buttons while a pipeline subprocess is running.

        Args:
            running: Whether a subprocess is currently active.

        Returns:
            None.
        """
        self.pipeline_running = running
        state = "disabled" if running else "normal"
        self.generate_button.configure(state=state)
        self.full_pipeline_button.configure(state=state)
        self.save_button.configure(state=state)

    def clear_log(self) -> None:
        """Open the detached log window and clear its content.

        Returns:
            None.
        """
        self.log_window.open()
        self.log_window.clear()

    def append_log(self, text: str) -> None:
        """Append one chunk of subprocess output to the detached log window.

        Args:
            text: Log text chunk.

        Returns:
            None.
        """
        self.log_window.append(text)

    def run_generated_pipeline(self) -> None:
        """Run only the final noisy/clean generation stage from the GUI.

        Returns:
            None. The subprocess is launched asynchronously.
        """
        self.start_pipeline(
            [
                sys.executable,
                "scripts/run_full_pipeline.py",
                "--skip-clean",
                "--skip-noise",
                "--reset-generated",
            ],
            "Final dataset generation",
        )

    def run_full_pipeline(self) -> None:
        """Run the full destructive pipeline after explicit user confirmation.

        Returns:
            None. The subprocess is launched asynchronously only after the user
            confirms cleanup and regeneration.
        """
        if not messagebox.askyesno(
            "Full pipeline",
            "The full pipeline deletes and recreates chunks, metadata, logs, and generated data. Continue?",
        ):
            return

        self.start_pipeline(
            [
                sys.executable,
                "scripts/run_full_pipeline.py",
                "--compile",
                "--reset-all",
            ],
            "Full pipeline",
        )

    def start_pipeline(self, command: list[str], title: str) -> None:
        """Validate and save config, then launch a pipeline command in the background.

        Args:
            command: Subprocess command list executed from the project root.
            title: Human-readable run title displayed in the log window.

        Returns:
            None.
        """
        if self.pipeline_running:
            return

        try:
            self.write_config()
        except Exception as exc:
            messagebox.showerror("Invalid configuration", f"Fix these parameters before running:\n{exc}")
            return

        self.log_window.open(title)
        self.clear_log()
        self.append_log(f"{title}\n")
        self.append_log("Command: " + " ".join(command) + "\n\n")
        self.status_var.set("Pipeline running...")
        self.set_running_state(True)

        thread = threading.Thread(target=self._run_process_worker, args=(command,), daemon=True)
        thread.start()
        self.app.after(100, self.poll_log_queue)

    def _run_process_worker(self, command: list[str]) -> None:
        """Stream subprocess output into a thread-safe queue for the GUI thread.

        Args:
            command: Subprocess command list executed from the project root.

        Returns:
            None. A ``None`` sentinel is added to ``log_queue`` when finished.
        """
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
                self.log_queue.put("\n[OK] Pipeline completed successfully.\n")
            else:
                self.log_queue.put(f"\n[ERROR] Pipeline exited with code {return_code}.\n")
        except Exception as exc:
            self.log_queue.put(f"\n[ERROR] Unable to start pipeline: {exc}\n")
        finally:
            self.log_queue.put(None)

    def poll_log_queue(self) -> None:
        """Move queued subprocess output into the log window from the GUI thread.

        Returns:
            None. Reschedules itself until the worker posts the finish sentinel.
        """
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
            self.status_var.set("Ready")
        else:
            self.app.after(100, self.poll_log_queue)


def main() -> None:
    """Launch the dataset configuration editor.

    Returns:
        None.
    """
    ModernConfigEditor().run()


if __name__ == "__main__":
    main()
