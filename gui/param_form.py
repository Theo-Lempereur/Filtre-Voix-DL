"""Formulaire d'hyperparamètres généré à partir de ``param_spec.PARAM_SPECS``.

Particularité du couple `resume` + `run_id` :
  - Si `resume` est vide → `run_id` est un QLineEdit (saisie libre, nouveau run).
  - Si `resume` est "last" / "best" / "best_si_sdri" → `run_id` devient un
    QComboBox listant uniquement les runs qui ont ce checkpoint sur disque.
    Quand un run y est sélectionné, le formulaire se remplit automatiquement
    avec la config sauvegardée dans ce checkpoint (lecture via
    ``ckpt_io.peek_checkpoint_config``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .param_spec import PARAM_SPECS, ParamKind, ParamSpec, by_name


# Mapping des clés de la config sauvegardée (cf. src/train._resolve_config)
# vers les noms de paramètres GUI. `use_wandb` est inclus parce qu'il est
# persisté ; les autres clés (weight_decay, lr_patience, etc.) ne sont pas
# exposées dans le formulaire et sont donc ignorées.
_CKPT_KEY_TO_FORM = {
    "num_epochs":    "epochs",
    "batch_size":    "batch_size",
    "lr":            "lr",
    "num_workers":   "num_workers",
    "base_channels": "base_channels",
    "seed":          "seed",
    "use_wandb":     "use_wandb",
}


def _make_widget(spec: ParamSpec) -> QWidget:
    kind: ParamKind = spec.kind
    w: QWidget
    if kind == "str":
        w = QLineEdit()
        if spec.default:
            w.setText(str(spec.default))
    elif kind == "int":
        sb = QSpinBox()
        if spec.minimum is not None:
            sb.setMinimum(int(spec.minimum))
        if spec.maximum is not None:
            sb.setMaximum(int(spec.maximum))
        if spec.step is not None:
            sb.setSingleStep(int(spec.step))
        sb.setValue(int(spec.default) if spec.default is not None else 0)
        w = sb
    elif kind == "float":
        dsb = QDoubleSpinBox()
        dsb.setDecimals(max(spec.decimals, 6))
        if spec.minimum is not None:
            dsb.setMinimum(float(spec.minimum))
        if spec.maximum is not None:
            dsb.setMaximum(float(spec.maximum))
        if spec.step is not None:
            dsb.setSingleStep(float(spec.step))
        dsb.setValue(float(spec.default) if spec.default is not None else 0.0)
        w = dsb
    elif kind == "bool":
        cb = QCheckBox()
        cb.setChecked(bool(spec.default))
        w = cb
    elif kind == "choice":
        cmb = QComboBox()
        for c in spec.choices:
            cmb.addItem(c if c else "(aucune)")
        try:
            idx = spec.choices.index(spec.default)
        except ValueError:
            idx = 0
        cmb.setCurrentIndex(idx)
        w = cmb
    else:
        w = QLabel(f"?? type inconnu : {kind}")
    w.setToolTip(spec.help)
    return w


def _list_runs_with_checkpoint(resume_mode: str) -> list[str]:
    """Énumère les run_ids ayant `<resume_mode>.pt` sur disque.

    Lit dynamiquement ``cfg.CHECKPOINTS`` à chaque appel (au cas où Drive
    Desktop viendrait d'être monté). Retourne la liste triée par mtime
    décroissant (le plus récent en premier) pour mettre le run le plus
    probable en tête.
    """
    try:
        # Import local : on ne veut pas faire crasher l'import de la GUI si
        # `src.config` lève à l'init (drive desktop pas encore monté, etc.).
        from src import config as cfg
    except Exception:
        return []

    root = Path(cfg.CHECKPOINTS)
    if not root.is_dir():
        return []

    runs: list[tuple[str, float]] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.is_dir():
            continue
        ckpt = entry / f"{resume_mode}.pt"
        if ckpt.is_file():
            try:
                mtime = ckpt.stat().st_mtime
            except OSError:
                mtime = 0.0
            runs.append((entry.name, mtime))
    runs.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _mtime in runs]


def _read_ckpt_config(run_id: str, resume_mode: str) -> dict | None:
    """Lit la config du checkpoint. Retourne None en cas d'échec."""
    try:
        from src import config as cfg
        from src import checkpoint as ckpt_io
    except Exception:
        return None
    path = Path(cfg.CHECKPOINTS) / run_id / f"{resume_mode}.pt"
    if not path.is_file():
        return None
    try:
        return ckpt_io.peek_checkpoint_config(path)
    except Exception:
        return None


class ParamForm(QWidget):
    """Formulaire complet."""

    # Émis quand l'utilisateur sélectionne un run depuis le ComboBox de reprise.
    config_loaded_from_ckpt = Signal(str, str)   # run_id, resume_mode

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._widgets: dict[str, QWidget] = {}
        self._enable_widgets: dict[str, QCheckBox] = {}
        # Stack pour run_id : index 0 = QLineEdit (saisie libre),
        # index 1 = QComboBox (sélection d'un run existant).
        self._run_id_stack: QStackedWidget | None = None
        self._run_id_line: QLineEdit | None = None
        self._run_id_combo: QComboBox | None = None
        # Label affiché sous le combo quand la liste est vide.
        self._run_id_status: QLabel | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        main_box = QGroupBox("Paramètres d'entraînement")
        main_form = QFormLayout(main_box)
        main_form.setLabelAlignment(Qt.AlignRight)

        adv_box = QGroupBox("Avancé")
        adv_box.setCheckable(True)
        adv_box.setChecked(False)
        adv_form = QFormLayout(adv_box)
        adv_form.setLabelAlignment(Qt.AlignRight)

        for spec in PARAM_SPECS:
            target_form = adv_form if spec.advanced else main_form
            row_widget = self._build_row(spec)
            target_form.addRow(spec.label, row_widget)

        root.addWidget(main_box)
        root.addWidget(adv_box)
        root.addStretch(1)

        # Câblage : si resume change, on rebascule le widget de run_id.
        resume_combo = self._widgets.get("resume")
        if isinstance(resume_combo, QComboBox):
            resume_combo.currentIndexChanged.connect(self._on_resume_changed)
        # Premier affichage cohérent (mode "nouveau run").
        self._on_resume_changed()

    # ------------------------------------------------------------- build row

    def _build_row(self, spec: ParamSpec) -> QWidget:
        if spec.name == "run_id":
            return self._build_run_id_row(spec)

        widget = _make_widget(spec)
        self._widgets[spec.name] = widget

        container = QFrame()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        if spec.optional:
            row = QHBoxLayout()
            enable = QCheckBox("activer")
            enable.setChecked(False)
            widget.setEnabled(False)
            enable.toggled.connect(widget.setEnabled)
            self._enable_widgets[spec.name] = enable
            row.addWidget(enable)
            row.addWidget(widget, 1)
            line = QFrame()
            line.setLayout(row)
            col.addWidget(line)
        else:
            col.addWidget(widget)

        if spec.help:
            help_label = QLabel(spec.help)
            help_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
            help_label.setWordWrap(True)
            col.addWidget(help_label)
        return container

    def _build_run_id_row(self, spec: ParamSpec) -> QWidget:
        """Construit la ligne `run_id` avec son stack (QLineEdit ↔ QComboBox)."""
        # Widget 0 : saisie libre
        line = QLineEdit()
        if spec.default:
            line.setText(str(spec.default))
        line.setToolTip(spec.help)
        # Widget 1 : sélection parmi les runs existants
        combo = QComboBox()
        combo.setToolTip(
            "Liste des runs ayant le checkpoint sélectionné dans 'Reprise'. "
            "Le formulaire se remplit automatiquement avec la config du run choisi."
        )
        combo.setEditable(False)
        combo.currentIndexChanged.connect(self._on_run_id_combo_changed)

        stack = QStackedWidget()
        stack.addWidget(line)   # index 0
        stack.addWidget(combo)  # index 1

        self._run_id_stack = stack
        self._run_id_line = line
        self._run_id_combo = combo
        self._widgets[spec.name] = stack  # exposé via le stack ; lectures via helpers

        container = QFrame()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        col.addWidget(stack)

        status = QLabel("")
        status.setStyleSheet("color: #c44; font-size: 10px; font-style: italic;")
        status.setWordWrap(True)
        status.hide()
        col.addWidget(status)
        self._run_id_status = status

        help_label = QLabel(spec.help)
        help_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        help_label.setWordWrap(True)
        col.addWidget(help_label)
        return container

    # ----------------------------------------------------- resume / run_id

    def _resume_mode(self) -> str:
        w = self._widgets.get("resume")
        if not isinstance(w, QComboBox):
            return ""
        idx = w.currentIndex()
        choices = by_name("resume").choices
        return choices[idx] if 0 <= idx < len(choices) else ""

    def _on_resume_changed(self) -> None:
        mode = self._resume_mode()
        if not mode:
            # Mode "nouveau run" : on remet le QLineEdit en avant.
            if self._run_id_stack:
                self._run_id_stack.setCurrentIndex(0)
            if self._run_id_status:
                self._run_id_status.hide()
            return
        # Mode "reprise" : on peuple le combobox avec les runs disponibles.
        runs = _list_runs_with_checkpoint(mode)
        if self._run_id_combo:
            self._run_id_combo.blockSignals(True)
            self._run_id_combo.clear()
            for r in runs:
                self._run_id_combo.addItem(r)
            self._run_id_combo.blockSignals(False)
        if self._run_id_stack:
            self._run_id_stack.setCurrentIndex(1)
        if self._run_id_status:
            if not runs:
                self._run_id_status.setText(
                    f"Aucun run avec un checkpoint '{mode}.pt' trouvé dans checkpoints/."
                )
                self._run_id_status.show()
            else:
                self._run_id_status.hide()
        # Si la liste n'est pas vide, on déclenche manuellement l'auto-fill
        # pour le premier élément (qui vient d'être sélectionné par le clear/add).
        if runs:
            self._on_run_id_combo_changed()

    def _on_run_id_combo_changed(self) -> None:
        mode = self._resume_mode()
        if not mode or not self._run_id_combo:
            return
        run_id = self._run_id_combo.currentText()
        if not run_id:
            return
        ckpt_cfg = _read_ckpt_config(run_id, mode)
        if not ckpt_cfg:
            if self._run_id_status:
                self._run_id_status.setText(
                    f"Impossible de lire la config de {run_id}/{mode}.pt"
                )
                self._run_id_status.show()
            return
        if self._run_id_status:
            self._run_id_status.hide()
        self._apply_ckpt_config(ckpt_cfg)
        self.config_loaded_from_ckpt.emit(run_id, mode)

    def _apply_ckpt_config(self, ckpt_cfg: dict) -> None:
        """Applique les valeurs du checkpoint aux widgets correspondants."""
        for ckpt_key, form_name in _CKPT_KEY_TO_FORM.items():
            if ckpt_key not in ckpt_cfg:
                continue
            self._set_widget_value(form_name, ckpt_cfg[ckpt_key])

    def _set_widget_value(self, name: str, value: Any) -> None:
        spec_obj = by_name(name)
        w = self._widgets.get(name)
        if w is None:
            return
        # On bloque les signaux le temps de l'écriture pour éviter qu'un
        # widget propage un changement (genre rechargement en cascade).
        w.blockSignals(True)
        try:
            if spec_obj.kind == "int" and isinstance(w, QSpinBox):
                try:
                    w.setValue(int(value))
                except (TypeError, ValueError):
                    pass
            elif spec_obj.kind == "float" and isinstance(w, QDoubleSpinBox):
                try:
                    w.setValue(float(value))
                except (TypeError, ValueError):
                    pass
            elif spec_obj.kind == "bool" and isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif spec_obj.kind == "choice" and isinstance(w, QComboBox):
                try:
                    idx = spec_obj.choices.index(str(value))
                    w.setCurrentIndex(idx)
                except ValueError:
                    pass
            elif spec_obj.kind == "str" and isinstance(w, QLineEdit):
                w.setText(str(value))
        finally:
            w.blockSignals(False)

    # --------------------------------------------------------------- read

    def _value(self, spec: ParamSpec) -> Any:
        if spec.name == "run_id":
            # Lecture spéciale : selon le stack actif.
            if self._run_id_stack and self._run_id_stack.currentIndex() == 1:
                return self._run_id_combo.currentText() if self._run_id_combo else ""
            return self._run_id_line.text().strip() if self._run_id_line else ""

        w = self._widgets[spec.name]
        if spec.kind == "str":
            return w.text().strip()  # type: ignore[union-attr]
        if spec.kind == "int":
            return int(w.value())     # type: ignore[union-attr]
        if spec.kind == "float":
            return float(w.value())   # type: ignore[union-attr]
        if spec.kind == "bool":
            return bool(w.isChecked())  # type: ignore[union-attr]
        if spec.kind == "choice":
            idx = w.currentIndex()    # type: ignore[union-attr]
            return spec.choices[idx]
        return None

    def values(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for spec in PARAM_SPECS:
            if spec.optional and not self._enable_widgets[spec.name].isChecked():
                continue
            out[spec.name] = self._value(spec)
        return out

    def to_cli_args(self) -> list[str]:
        args: list[str] = []
        for spec in PARAM_SPECS:
            if spec.optional and not self._enable_widgets[spec.name].isChecked():
                continue
            val = self._value(spec)
            if spec.kind == "bool":
                if spec.inverted:
                    if not val:
                        args.append(spec.cli_flag)
                else:
                    if val:
                        args.append(spec.cli_flag)
            elif spec.kind == "choice":
                if val:
                    args += [spec.cli_flag, str(val)]
            elif spec.kind == "str":
                if val:
                    args += [spec.cli_flag, str(val)]
            else:
                args += [spec.cli_flag, str(val)]
        return args

    def validate(self) -> list[str]:
        errors: list[str] = []
        run_id = self.get_run_id()
        mode = self._resume_mode()
        if not run_id:
            if mode:
                errors.append(
                    f"Aucun run avec un checkpoint '{mode}.pt' n'est disponible "
                    "pour reprendre. Choisis un autre mode de reprise ou "
                    "lance un nouveau run."
                )
            else:
                errors.append("Run ID est obligatoire.")
        elif not mode and any(c in run_id for c in r' /\:*?"<>|'):
            errors.append("Run ID contient des caractères invalides pour un nom de dossier.")
        return errors

    # ----------------------------------------------------- helpers publics

    def get_run_id(self) -> str:
        return str(self._value(by_name("run_id")))

    def set_run_id(self, value: str) -> None:
        """Pré-remplit le run_id (no-op en mode reprise — pas de sens d'écraser
        une sélection dans la liste déroulante)."""
        if self._resume_mode():
            return
        if self._run_id_line is not None:
            self._run_id_line.setText(value)

    def is_resume_mode(self) -> bool:
        return bool(self._resume_mode())
