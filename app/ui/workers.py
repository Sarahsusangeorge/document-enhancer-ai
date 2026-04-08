"""QThread workers for offloading AI-heavy operations to background threads.

Provides QRunnable-based workers with signal support for progress reporting,
cancellation, and completion notification.
"""

import logging
import traceback
from typing import List, Optional

import numpy as np

try:
    from PySide6.QtCore import QObject, QRunnable, Signal, Slot
except ImportError:
    pass

from app.core.pipeline import Pipeline, PipelineConfig
from app.models.document import Document, PipelineStage

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Signals emitted by background workers.

    Attributes:
        stage_progress: (stage_value: str, percent: float, message: str)
        finished: emitted with the resulting Document on success
        error: emitted with a traceback string on failure
        cancelled: emitted when the user cancels the operation
    """

    stage_progress = Signal(str, float, str)
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()


class PipelineWorker(QRunnable):
    """Runs the full enhancement pipeline in a background thread."""

    def __init__(self, images: List[np.ndarray], document: Document,
                 config: Optional[PipelineConfig] = None):
        super().__init__()
        self.signals = WorkerSignals()
        self.images = images
        self.document = document
        self.config = config or PipelineConfig()
        self._pipeline: Optional[Pipeline] = None
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            self._pipeline = Pipeline(self.config)
            self._pipeline.set_progress_callback(self._on_progress)

            if len(self.images) == 1:
                result = self._pipeline.process_image(
                    self.images[0], self.document,
                )
            else:
                result = self._pipeline.process_images(
                    self.images, self.document,
                )

            if self.document.processing_state.has_errors:
                errors = self.document.processing_state.errors
                last = errors[-1] if errors else {}
                if last.get("error") == "Cancelled by user":
                    self.signals.cancelled.emit()
                    return

                ocr_errors = [
                    e for e in errors
                    if e.get("stage") in ("preprocessing", "ocr")
                ]
                if ocr_errors:
                    msg = ocr_errors[-1].get("error", "Unknown OCR error")
                    self.signals.error.emit(
                        f"OCR failed: {msg}\n\n"
                        "Make sure Tesseract OCR is installed and its path "
                        "is configured in Settings."
                    )
                    return

            self.signals.finished.emit(result)

        except InterruptedError:
            self.signals.cancelled.emit()
        except Exception:
            tb = traceback.format_exc()
            logger.error("PipelineWorker error:\n%s", tb)
            self.signals.error.emit(tb)

    def cancel(self) -> None:
        if self._pipeline is not None:
            self._pipeline.cancel()

    def _on_progress(self, stage: PipelineStage, percent: float,
                     message: str) -> None:
        self.signals.stage_progress.emit(stage.value, percent, message)


class ExportWorker(QRunnable):
    """Exports a document to DOCX or PDF in a background thread."""

    def __init__(self, document: Document, output_path: str,
                 doc_type: Optional[str] = None):
        super().__init__()
        self.signals = WorkerSignals()
        self.document = document
        self.output_path = output_path
        self.doc_type = doc_type
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            from app.utils.exporter import DocumentExporter

            self.signals.stage_progress.emit("export", 0, "Exporting...")
            exporter = DocumentExporter()
            exporter.export(
                self.document, self.output_path, doc_type=self.doc_type,
            )
            self.signals.stage_progress.emit("export", 100, "Export complete")
            self.signals.finished.emit(self.output_path)

        except Exception:
            tb = traceback.format_exc()
            logger.error("ExportWorker error:\n%s", tb)
            self.signals.error.emit(tb)
