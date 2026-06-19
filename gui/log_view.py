"""Vue de logs append-only avec autoscroll."""
from __future__ import annotations

from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


class LogView(QWidget):
    """QPlainTextEdit configuré en append-only avec auto-scroll et bouton clear."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._edit = QPlainTextEdit()
        self._edit.setReadOnly(True)
        self._edit.setMaximumBlockCount(10_000)  # garde-fou mémoire
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(9)
        self._edit.setFont(font)
        root.addWidget(self._edit, 1)

        controls = QHBoxLayout()
        controls.addStretch(1)
        self.btn_clear = QPushButton("Effacer")
        self.btn_clear.clicked.connect(self._edit.clear)
        controls.addWidget(self.btn_clear)
        root.addLayout(controls)

    def append(self, text: str) -> None:
        # On retire un éventuel '\n' final pour ne pas doubler les lignes
        # (appendPlainText ajoute déjà son propre saut).
        if text.endswith("\n"):
            text = text[:-1]
        if not text:
            return
        self._edit.appendPlainText(text)
        # Auto-scroll si la barre est déjà en bas.
        sb = self._edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self) -> None:
        self._edit.clear()
