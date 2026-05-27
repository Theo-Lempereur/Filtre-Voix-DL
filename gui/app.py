"""Fenêtre principale de la GUI d'orchestration des entraînements."""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .log_view import LogView
from .param_form import ParamForm
from .queue_widget import QueueWidget
from .runner import TrainingRunner


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Filtre-Voix-DL — Orchestrateur d'entraînements")
        self.resize(1500, 900)

        self.runner = TrainingRunner(self)

        # --- Colonne gauche : formulaire ---
        self.form = ParamForm()
        form_scroll = QScrollArea()
        form_scroll.setWidget(self.form)
        form_scroll.setWidgetResizable(True)
        form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.btn_enqueue = QPushButton("➕ Ajouter à la queue")
        self.btn_enqueue.setStyleSheet("padding: 8px; font-weight: bold;")
        self.btn_enqueue.clicked.connect(self._on_enqueue_clicked)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.addWidget(form_scroll, 1)
        left_layout.addWidget(self.btn_enqueue)

        # --- Colonne centre : queue + contrôles ---
        self.queue_view = QueueWidget()
        self.queue_view.move_requested.connect(self.runner.move)
        self.queue_view.remove_requested.connect(self.runner.remove)
        self.queue_view.reordered_by_drag.connect(self.runner.reorder_by_run_ids)

        self.lbl_state = QLabel("État : idle")
        self.lbl_state.setStyleSheet("font-weight: bold; padding: 4px;")

        self.btn_stop = QPushButton("⏹ Stop run courant")
        self.btn_stop.setStyleSheet("padding: 8px; background-color: #e63946; color: white;")
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_stop.setEnabled(False)

        self.btn_resume = QPushButton("▶ Reprendre la queue")
        self.btn_resume.setStyleSheet("padding: 8px;")
        self.btn_resume.clicked.connect(self._on_resume_clicked)

        self.btn_clear = QPushButton("Vider la queue (sauf run courant)")
        self.btn_clear.clicked.connect(self._on_clear_clicked)

        self.btn_force_kill = QPushButton("Forcer kill (dernier recours)")
        self.btn_force_kill.setStyleSheet("color: #b00020;")
        self.btn_force_kill.clicked.connect(self._on_force_kill_clicked)
        self.btn_force_kill.setEnabled(False)

        queue_box = QGroupBox("Queue d'entraînements")
        queue_layout = QVBoxLayout(queue_box)
        queue_layout.addWidget(self.lbl_state)
        queue_layout.addWidget(self.queue_view, 1)
        controls = QHBoxLayout()
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_resume)
        queue_layout.addLayout(controls)
        queue_layout.addWidget(self.btn_clear)
        queue_layout.addWidget(self.btn_force_kill)

        center = QWidget()
        c_layout = QVBoxLayout(center)
        c_layout.setContentsMargins(8, 8, 8, 8)
        c_layout.addWidget(queue_box, 1)

        # --- Colonne droite : logs ---
        self.log_view = LogView()
        log_box = QGroupBox("Logs du run en cours")
        log_layout = QVBoxLayout(log_box)
        log_layout.addWidget(self.log_view)
        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.setContentsMargins(8, 8, 8, 8)
        r_layout.addWidget(log_box, 1)

        # --- Splitter principal ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 4)
        self.setCentralWidget(splitter)

        # --- Status bar ---
        self.setStatusBar(QStatusBar())
        self._refresh_status_bar()

        # --- Connexions runner ---
        self.runner.log_line.connect(self.log_view.append)
        self.runner.state_changed.connect(self._on_state_changed)
        self.runner.queue_changed.connect(self._refresh_queue_view)
        self.runner.run_started.connect(self._on_run_started)
        self.runner.run_finished.connect(self._on_run_finished)
        self.runner.error.connect(self._on_runner_error)

        self._refresh_queue_view()

    # ---------------------------------------------------------------- slots

    def _on_enqueue_clicked(self) -> None:
        errors = self.form.validate()
        run_id = self.form.get_run_id()
        if not errors and self.runner.has_run_id(run_id):
            errors.append(f"Run ID '{run_id}' déjà présent dans la queue.")
        if errors:
            QMessageBox.warning(self, "Validation", "\n".join(errors))
            return
        args = self.form.to_cli_args()
        values = self.form.values()
        self.runner.enqueue(run_id, args, values)
        # Pré-remplit un nouveau run_id (incrément simple) pour faciliter l'ajout
        # d'une série d'expérimentations.
        self.form.set_run_id(_bump_run_id(run_id))
        # Démarre tout de suite si on est idle et qu'il n'y avait rien.
        if self.runner.state == "idle":
            self.runner.start_next_if_idle()

    def _on_stop_clicked(self) -> None:
        self.runner.request_stop_current()

    def _on_resume_clicked(self) -> None:
        self.runner.resume_queue()

    def _on_clear_clicked(self) -> None:
        if not self.runner.jobs_snapshot():
            return
        reply = QMessageBox.question(
            self, "Vider la queue",
            "Supprimer tous les jobs en attente ? (le run en cours n'est pas touché)",
        )
        if reply == QMessageBox.Yes:
            self.runner.clear_queue()

    def _on_force_kill_clicked(self) -> None:
        reply = QMessageBox.warning(
            self, "Force kill",
            "Tuer le process maintenant ? Le checkpoint en cours d'écriture peut être corrompu.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.runner.force_kill_current()

    def _on_state_changed(self, state: str) -> None:
        self.lbl_state.setText(f"État : {state}")
        self.btn_stop.setEnabled(state == "running")
        self.btn_force_kill.setEnabled(state == "stopping")
        self.btn_resume.setEnabled(state in ("paused", "idle"))
        self._refresh_status_bar()

    def _on_run_started(self, run_id: str) -> None:
        self.log_view.clear()
        self.log_view.append(f"[gui] démarrage du run '{run_id}'…")
        self._refresh_status_bar()

    def _on_run_finished(self, run_id: str, exit_code: int) -> None:
        self.log_view.append(f"[gui] run '{run_id}' fini (exit={exit_code}).")
        self._refresh_status_bar()

    def _on_runner_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Runner", msg)

    def _refresh_queue_view(self) -> None:
        jobs = self.runner.jobs_snapshot()
        self.queue_view.set_jobs(jobs, self.runner.running_index())
        self._refresh_status_bar()

    def _refresh_status_bar(self) -> None:
        jobs = self.runner.jobs_snapshot()
        running = jobs[0][0] if self.runner.running_index() == 0 and jobs else "—"
        waiting = max(0, len(jobs) - (1 if self.runner.running_index() is not None else 0))
        self.statusBar().showMessage(
            f"Run en cours : {running}   •   Queue : {waiting} en attente"
        )

    # --------------------------------------------------------------- close

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.runner.is_running():
            reply = QMessageBox.question(
                self, "Fermer",
                "Un entraînement est en cours. Demander un stop propre puis fermer ? "
                "(le process continuera tant qu'il n'aura pas fini de sauvegarder)",
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        self.runner.shutdown()
        event.accept()


def _bump_run_id(prev: str) -> str:
    """Incrémente un suffixe numérique en fin de run_id, ou ajoute _2."""
    import re
    m = re.search(r"^(.*?)(\d+)$", prev)
    if m:
        base, num = m.group(1), int(m.group(2))
        return f"{base}{num + 1}"
    if prev:
        return f"{prev}_2"
    return ""


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()
