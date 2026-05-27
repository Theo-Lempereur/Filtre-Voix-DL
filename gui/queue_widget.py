"""Widget de queue : liste réordonnable avec boutons supprimer / monter / descendre.

Le premier item correspond au run en cours quand le runner tourne (figé,
non draggable). Les items suivants sont les jobs en attente.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _format_summary(run_id: str, values: dict[str, Any]) -> str:
    """Résumé court pour afficher dans la queue."""
    bits = [f"epochs={values.get('epochs')}", f"batch={values.get('batch_size')}",
            f"lr={values.get('lr'):.0e}" if isinstance(values.get('lr'), float) else ""]
    extra = []
    if values.get("resume"):
        extra.append(f"resume={values['resume']}")
    if values.get("base_channels") not in (None, 32):
        extra.append(f"ch={values['base_channels']}")
    summary = "  •  ".join([b for b in bits if b] + extra)
    return f"{run_id}  •  {summary}"


class QueueWidget(QWidget):
    """Liste de jobs en attente. Communique avec ``TrainingRunner`` via signaux."""

    move_requested = Signal(int, int)     # from_index, to_index
    remove_requested = Signal(int)        # index
    reordered_by_drag = Signal(list)      # new ordering of run_ids

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        root.addWidget(self._list, 1)

        controls = QHBoxLayout()
        self.btn_up = QPushButton("↑ Monter")
        self.btn_down = QPushButton("↓ Descendre")
        self.btn_remove = QPushButton("✕ Supprimer")
        controls.addWidget(self.btn_up)
        controls.addWidget(self.btn_down)
        controls.addWidget(self.btn_remove)
        root.addLayout(controls)

        self.btn_up.clicked.connect(self._move_up)
        self.btn_down.clicked.connect(self._move_down)
        self.btn_remove.clicked.connect(self._remove_selected)

        self._running_index: int | None = None

    # ------------------------------------------------------------------ API

    def set_jobs(self, jobs: list[tuple[str, dict[str, Any]]], running_index: int | None) -> None:
        """Reconstruit complètement la liste depuis l'état du runner.

        ``jobs`` est une liste ordonnée [(run_id, values_snapshot), ...]
        ``running_index`` indique quel item est en cours (None si idle).
        """
        # On bloque les signaux le temps de reconstruire pour éviter une
        # cascade `rowsMoved` parasite.
        self._list.blockSignals(True)
        self._list.clear()
        for i, (run_id, values) in enumerate(jobs):
            text = _format_summary(run_id, values)
            if i == running_index:
                text = f"▶ EN COURS  —  {text}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, run_id)
            if i == running_index:
                item.setBackground(QColor("#cfe8ff"))
                # Empêche le drag et la sélection pour le run actif.
                flags = item.flags()
                flags &= ~Qt.ItemIsDragEnabled
                item.setFlags(flags)
            self._list.addItem(item)
        self._running_index = running_index
        self._list.blockSignals(False)
        self._update_buttons()

    # --------------------------------------------------------- interactions

    def _selected_index(self) -> int | None:
        rows = self._list.selectedIndexes()
        if not rows:
            return None
        return rows[0].row()

    def _move_up(self) -> None:
        i = self._selected_index()
        if i is None:
            return
        # On ne peut pas franchir l'index du run en cours.
        min_movable = (self._running_index + 1) if self._running_index is not None else 0
        if i <= min_movable:
            return
        self.move_requested.emit(i, i - 1)

    def _move_down(self) -> None:
        i = self._selected_index()
        if i is None:
            return
        if i == self._running_index:
            return
        if i >= self._list.count() - 1:
            return
        self.move_requested.emit(i, i + 1)

    def _remove_selected(self) -> None:
        i = self._selected_index()
        if i is None:
            return
        if i == self._running_index:
            return  # impossible : pas dans la sélection draggable, mais on garde la garde
        self.remove_requested.emit(i)

    def _on_rows_moved(self, *_args) -> None:
        """Émis par le model après un drag-drop réussi.

        On lit la nouvelle séquence de run_ids et on demande au runner de
        se resynchroniser.
        """
        ordering: list[str] = []
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it is None:
                continue
            rid = it.data(Qt.UserRole)
            if rid is None:
                continue
            ordering.append(str(rid))
        self.reordered_by_drag.emit(ordering)

    def _update_buttons(self) -> None:
        i = self._selected_index()
        can_remove = i is not None and i != self._running_index
        self.btn_remove.setEnabled(can_remove)
        self.btn_up.setEnabled(can_remove)
        self.btn_down.setEnabled(can_remove)
