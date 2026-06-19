"""Contrôleur d'entraînement : QProcess + queue en mémoire.

Lance ``scripts/train_local.py`` en sous-processus, gère la queue de jobs en
attente, expose l'API pour ajouter / supprimer / réordonner / stopper.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_local.py"


class TrainingRunner(QObject):
    """Orchestrateur de runs séquentiels.

    États possibles :
      - "idle"     : aucun process en cours
      - "running"  : un process tourne
      - "stopping" : stop demandé, on attend que le process se termine
      - "paused"   : run précédent terminé via stop, on ne démarre PAS le suivant
                     automatiquement (l'utilisateur doit cliquer "Reprendre").
    """

    log_line       = Signal(str)
    state_changed  = Signal(str)
    run_started    = Signal(str)         # run_id
    run_finished   = Signal(str, int)    # run_id, exit_code
    queue_changed  = Signal()
    error          = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: deque[tuple[str, list[str], dict[str, Any]]] = deque()
        self._state: str = "idle"
        self._current: tuple[str, list[str], dict[str, Any]] | None = None
        self._process: QProcess | None = None

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, new: str) -> None:
        if self._state != new:
            self._state = new
            self.state_changed.emit(new)

    def is_running(self) -> bool:
        return self._state in ("running", "stopping")

    # ----------------------------------------------------------- queue ops

    def jobs_snapshot(self) -> list[tuple[str, dict[str, Any]]]:
        """Renvoie la liste complète des jobs visibles dans la GUI.

        Inclut le run en cours en tête de liste (s'il y en a un).
        """
        out: list[tuple[str, dict[str, Any]]] = []
        if self._current is not None:
            out.append((self._current[0], self._current[2]))
        for rid, _args, vals in self._queue:
            out.append((rid, vals))
        return out

    def running_index(self) -> int | None:
        return 0 if self._current is not None else None

    def has_run_id(self, run_id: str) -> bool:
        if self._current is not None and self._current[0] == run_id:
            return True
        return any(rid == run_id for rid, _a, _v in self._queue)

    def enqueue(self, run_id: str, cli_args: list[str], values: dict[str, Any]) -> None:
        self._queue.append((run_id, cli_args, values))
        self.queue_changed.emit()

    def remove(self, index: int) -> None:
        """Supprime l'item à `index` (numérotation incluant le run courant en 0)."""
        offset = 1 if self._current is not None else 0
        q_index = index - offset
        if q_index < 0 or q_index >= len(self._queue):
            return
        items = list(self._queue)
        del items[q_index]
        self._queue = deque(items)
        self.queue_changed.emit()

    def move(self, from_index: int, to_index: int) -> None:
        offset = 1 if self._current is not None else 0
        f = from_index - offset
        t = to_index - offset
        if f < 0 or f >= len(self._queue):
            return
        t = max(0, min(t, len(self._queue) - 1))
        items = list(self._queue)
        item = items.pop(f)
        items.insert(t, item)
        self._queue = deque(items)
        self.queue_changed.emit()

    def reorder_by_run_ids(self, ordering: list[str]) -> None:
        """Resynchronise la queue après un drag-drop dans la vue.

        ``ordering`` est la séquence complète de run_ids vue par la GUI
        (incluant éventuellement le run en cours en tête, qu'on ignore).
        """
        if self._current is not None and ordering and ordering[0] == self._current[0]:
            ordering = ordering[1:]
        # On reconstruit la deque en respectant le nouvel ordre.
        index: dict[str, tuple[str, list[str], dict[str, Any]]] = {
            rid: (rid, args, vals) for rid, args, vals in self._queue
        }
        new_items: list[tuple[str, list[str], dict[str, Any]]] = []
        for rid in ordering:
            if rid in index:
                new_items.append(index.pop(rid))
        # Les éventuels orphelins (ne devrait pas arriver) sont remis à la fin.
        new_items.extend(index.values())
        self._queue = deque(new_items)
        self.queue_changed.emit()

    def clear_queue(self) -> None:
        """Vide les jobs en attente. Ne touche pas au run en cours."""
        self._queue.clear()
        self.queue_changed.emit()

    # ----------------------------------------------------------- run ops

    def start_next_if_idle(self) -> bool:
        """Démarre le prochain job si on est idle. Retourne True si démarré."""
        if self._state != "idle":
            return False
        if not self._queue:
            return False
        run_id, args, values = self._queue.popleft()
        self._current = (run_id, args, values)
        self._launch_process(run_id, args)
        self.queue_changed.emit()
        return True

    def resume_queue(self) -> bool:
        """Sort de l'état 'paused' et tente de démarrer le prochain job."""
        if self._state == "paused":
            self._set_state("idle")
            return self.start_next_if_idle()
        return self.start_next_if_idle()

    def request_stop_current(self) -> None:
        """Écrit le flag stop. Le training détectera et sortira proprement."""
        if self._current is None or self._state != "running":
            return
        run_id = self._current[0]
        try:
            # Import local pour ne pas tirer src.config au chargement de la GUI
            # sur des machines où Drive Desktop n'est pas (encore) monté.
            from src import stop_signal
            stop_signal.request_stop(run_id)
        except Exception as e:
            self.error.emit(f"Impossible d'écrire le flag stop : {e}")
            return
        self._set_state("stopping")
        self.log_line.emit(f"[gui] stop demandé pour {run_id}, en attente de l'arrêt propre…")

    def force_kill_current(self) -> None:
        """Dernier recours : kill -9 du process. À n'utiliser que si stop bloque."""
        if self._process is None:
            return
        self.log_line.emit("[gui] force kill du process en cours…")
        self._process.kill()

    def shutdown(self) -> None:
        """À appeler quand la fenêtre se ferme."""
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            # Demande propre puis force après 30 s.
            self.request_stop_current()
            self._process.waitForFinished(30_000)
            if self._process.state() != QProcess.NotRunning:
                self._process.kill()
                self._process.waitForFinished(2000)

    # ---------------------------------------------------------- internals

    def _launch_process(self, run_id: str, cli_args: list[str]) -> None:
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setWorkingDirectory(str(REPO_ROOT))
        # IMPORTANT : par défaut, `proc.processEnvironment()` retourne un dict
        # VIDE — il faut explicitement partir de `systemEnvironment()` pour
        # hériter de USERPROFILE / HOME / PATH / etc. Sans ça, `Path.home()`
        # côté child lève RuntimeError("Could not determine home directory").
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(env)

        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)

        python = sys.executable
        args = [str(TRAIN_SCRIPT), *cli_args]
        self.log_line.emit(f"[gui] $ {python} {' '.join(args)}")
        proc.start(python, args)

        self._process = proc
        self._set_state("running")
        self.run_started.emit(run_id)

    def _on_stdout(self) -> None:
        if self._process is None:
            return
        raw = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in raw.splitlines():
            self.log_line.emit(line)

    def _on_error(self, err) -> None:
        # err est un QProcess.ProcessError
        self.error.emit(f"QProcess error : {err}")

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        run_id = self._current[0] if self._current else "?"
        self.log_line.emit(f"[gui] process terminé (exit={exit_code}).")
        was_stopping = self._state == "stopping"
        self._current = None
        self._process = None
        self.run_finished.emit(run_id, int(exit_code))
        if was_stopping:
            # On ne chaîne PAS sur le suivant : la queue reste en pause.
            self._set_state("paused")
            self.log_line.emit("[gui] queue en pause après stop. Clique 'Reprendre la queue' pour la relancer.")
            self.queue_changed.emit()
        else:
            self._set_state("idle")
            self.queue_changed.emit()
            if self._queue:
                self.start_next_if_idle()
