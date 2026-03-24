# AI Document Enhancement System

A desktop application built with PySide6 that transforms scanned document images into polished, enhanced digital text. It chains OCR extraction, grammar correction, readability optimization, summarization, plagiarism checking, paraphrasing, and template-based formatting into a single configurable pipeline.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Launching the Application](#launching-the-application)
5. [Usage](#usage)
6. [Pipeline Stages](#pipeline-stages)
7. [Configuration](#configuration)
8. [Project Structure](#project-structure)
9. [Testing](#testing)
10. [Development Setup](#development-setup)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| **OCR Extraction** | Tesseract-based text extraction from scanned images and multi-page PDFs, with handwriting support and per-word confidence scores |
| **Grammar Enhancement** | Automated grammar and spelling correction via LanguageTool with before/after diff tracking |
| **Readability Optimization** | Flesch-Kincaid scoring and rule-based simplification to reach a target grade level |
| **Summarization** | Extractive (LSA via sumy) and abstractive (DistilBART via Hugging Face) summarization with key-point extraction |
| **Document Formatting** | Keyword-based type detection (essay, report, letter, notes, research paper) and python-docx template application |
| **Plagiarism Checking** | Local TF-IDF cosine similarity analysis with a pluggable API interface and similarity highlights |
| **Paraphrasing** | T5-based paraphrase generation with multiple suggestions for flagged passages |
| **Export** | DOCX export with template styling and PDF export via fpdf2, both with embedded metadata |
| **Dark / Light Themes** | Toggle between dark and light QSS stylesheets from the sidebar |
| **Threaded Processing** | All AI-heavy operations run in background QThread workers with progress reporting and cancellation |

---

## Prerequisites

### Python

Python **3.10 or newer** is required. Verify with:

```bash
python --version
```

### Tesseract OCR

Tesseract is used for text extraction. Install it and make sure the executable is accessible.

- **Windows** — Download the installer from <https://github.com/UB-Mannheim/tesseract/wiki> and run it. The default install path is `C:\Program Files\Tesseract-OCR`. You can either add that directory to your system `PATH` or configure the path inside the application's Settings page.
- **macOS** — `brew install tesseract`
- **Linux** — `sudo apt-get install tesseract-ocr`

Verify the installation:

```bash
tesseract --version
```

### Poppler (for PDF support)

`pdf2image` requires the Poppler library to convert PDF pages to images.

- **Windows** — Download a pre-built binary from <https://github.com/oschwartz10612/poppler-windows/releases>, extract it, and add the `bin/` folder to your system `PATH`.
- **macOS** — `brew install poppler`
- **Linux** — `sudo apt-get install poppler-utils`

### LanguageTool (optional — auto-downloaded)

The `language-tool-python` package downloads a local LanguageTool server automatically on first use. A working Java Runtime Environment (JRE 8+) must be installed for this to function. If Java is not available the grammar stage is skipped gracefully.

---

## Installation

Follow these steps to install the application and build a standalone Windows executable (`.exe`) that you can launch by double-clicking — just like any other desktop app.

### Step 1 — Clone the Repository

```bash
git clone <repo-url>
cd SE_project
```

### Step 2 — Create a Virtual Environment and Install Dependencies

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows (PowerShell / CMD)
# source venv/bin/activate   # macOS / Linux

# Install all dependencies
pip install -r requirements.txt
```

> **Tip:** On machines without a GPU, PyTorch will install the CPU-only variant automatically. The summarizer and paraphraser models (DistilBART, T5) run on CPU by default.

### Step 3 — Build the Windows Executable

The project includes a PyInstaller spec file (`ai_document_enhancer.spec`) pre-configured to bundle all resources, the app icon, hidden imports, and runtime hooks. Run:

```bash
pyinstaller ai_document_enhancer.spec
```

This creates the application at:

```
dist\AIDocumentEnhancer\AIDocumentEnhancer.exe
```

The output is a folder (`dist\AIDocumentEnhancer\`) containing the `.exe` and all its supporting files. To distribute the app, zip and share the entire folder.

### Step 4 — Create a Desktop Shortcut (optional)

1. Navigate to `dist\AIDocumentEnhancer\` in File Explorer.
2. Right-click **AIDocumentEnhancer.exe** → **Show more options** → **Create shortcut**.
3. Move the shortcut to your Desktop or pin it to your Taskbar.

You can now launch the application by double-clicking the shortcut — no terminal or Python required.

> **Note:** Tesseract OCR and Poppler must still be installed on the machine (see [Prerequisites](#prerequisites)). On first run, the LanguageTool Java server and Hugging Face models (~750 MB) will download automatically if not already cached.

---

## Launching the Application

### Option A — Double-click the Executable (recommended)

After building, open `dist\AIDocumentEnhancer\AIDocumentEnhancer.exe` (or use the desktop shortcut you created). The application window will open directly.

### Option B — Run from Source (for development)

If you prefer to run from source without building:

```bash
cd SE_project
venv\Scripts\activate
python main.py
```

The application opens in **dark theme** by default. Use the theme toggle in the sidebar to switch to light mode.

---

## Usage

### Workflow

1. **Home** — Dashboard showing a welcome screen and quick-start guidance.
2. **Upload** — Drag-and-drop or browse to select a scanned image (`.png`, `.jpg`, `.bmp`, `.tiff`) or a multi-page PDF. Thumbnail previews are generated for each page.
3. **Preview** — Side-by-side view of the original image and the OCR-extracted text. Low-confidence words are highlighted.
4. **Editor** — Manual text editor for correcting OCR mistakes before enhancement. Words with low confidence are visually flagged.
5. **Enhance** — Run grammar correction, readability optimization, and summarization. Each sub-stage reports progress in real time. Toggle individual stages on or off.
6. **Results** — View the final enhanced text, readability metrics, summary, and plagiarism score. Export to DOCX or PDF with template formatting.
7. **Settings** — Configure the Tesseract executable path, OCR language, summarization method (extractive/abstractive), target readability grade, similarity threshold, and model preferences.

### Keyboard Shortcut

| Action | Shortcut |
|--------|----------|
| Cancel running pipeline | **Esc** or the Cancel button in the progress bar |

---

## Pipeline Stages

The processing pipeline runs the following stages in order. Each stage can be individually enabled or disabled from the Settings page or the Enhance page controls.

```
Image → Preprocessing → OCR → Grammar → Readability → Summarization
       → Plagiarism Check → Paraphrasing (if flagged) → Formatting
```

| Stage | Module | What It Does |
|-------|--------|--------------|
| Preprocessing | `app/core/preprocessing.py` | Grayscale conversion, adaptive thresholding, skew correction (Hough transform), CLAHE contrast enhancement |
| OCR | `app/core/ocr_engine.py` | pytesseract wrapper supporting handwriting config, per-word confidence, and batch multi-page extraction |
| Grammar | `app/core/grammar_enhancer.py` | LanguageTool integration producing before/after diffs and correction counts |
| Readability | `app/core/readability_optimizer.py` | textstat Flesch-Kincaid scoring with rule-based sentence simplification |
| Summarization | `app/core/summarizer.py` | Extractive (sumy LSA) or abstractive (DistilBART) summary with configurable length |
| Plagiarism | `app/core/plagiarism_checker.py` | TF-IDF cosine similarity in local mode; pluggable API interface for external services |
| Paraphrasing | `app/core/paraphraser.py` | T5-based paraphrase generation for passages exceeding the similarity threshold |
| Formatting | `app/core/document_formatter.py` | Keyword-based document type detection and template application for DOCX output |

---

## Configuration

### Settings Page (in-app)

All runtime settings are persisted to a local SQLite database and restored on next launch.

| Setting | Description | Default |
|---------|-------------|---------|
| Tesseract Path | Path to the `tesseract` executable | System PATH |
| OCR Language | Tesseract language code (`eng`, `fra`, etc.) | `eng` |
| Summarization Method | `extractive` (LSA) or `abstractive` (DistilBART) | `extractive` |
| Summary Sentences | Number of sentences in the summary | `5` |
| Target Readability Grade | Flesch-Kincaid grade level target | `10.0` |
| Similarity Threshold | Plagiarism flagging threshold (0.0–1.0) | `0.7` |
| Paraphrase Suggestions | Number of paraphrase alternatives per flagged passage | `3` |

### Document Templates

Formatting templates live in `resources/templates/` as JSON files. Each template defines font, spacing, margins, alignment, and section structure. Bundled templates:

- **Essay** — Times New Roman, double-spaced, 0.5″ first-line indent
- **Report** — Calibri, 1.15 spacing, numbered headings
- **Letter** — Times New Roman, single-spaced, formal block layout
- **Notes** — Arial, compact single-spaced bullet-style
- **Research Paper** — Times New Roman, double-spaced, abstract + citations sections

To add a custom template, create a new JSON file in `resources/templates/` following the same schema.

---

## Project Structure

```
SE_project/
├── app/
│   ├── core/               # AI pipeline modules
│   │   ├── preprocessing.py
│   │   ├── ocr_engine.py
│   │   ├── grammar_enhancer.py
│   │   ├── readability_optimizer.py
│   │   ├── summarizer.py
│   │   ├── plagiarism_checker.py
│   │   ├── paraphraser.py
│   │   ├── document_formatter.py
│   │   └── pipeline.py         # End-to-end orchestrator
│   ├── database/
│   │   └── db_manager.py       # SQLite CRUD for documents, history, settings
│   ├── models/
│   │   └── document.py         # Document dataclass and enums
│   ├── ui/                     # PySide6 pages and widgets
│   │   ├── main_window.py
│   │   ├── home_page.py
│   │   ├── upload_page.py
│   │   ├── preview_page.py
│   │   ├── editor_page.py
│   │   ├── enhance_page.py
│   │   ├── results_page.py
│   │   ├── settings_page.py
│   │   ├── styles.py           # QSS dark/light themes
│   │   ├── widgets.py          # Reusable custom widgets
│   │   └── workers.py          # QThread workers and signals
│   └── utils/
│       ├── file_handler.py     # File I/O and PDF page extraction
│       └── exporter.py         # DOCX/PDF export with templates
├── resources/
│   ├── icons/                  # SVG sidebar and UI icons
│   └── templates/              # Document formatting templates (JSON)
├── tests/
│   ├── test_pipeline.py        # Unit tests for each core module
│   ├── test_ui.py              # PySide6 UI tests (pytest-qt)
│   ├── test_mutation.py        # Mutation tests (mutmut)
│   └── ...
├── .github/workflows/ci_cd.yml # GitHub Actions CI/CD pipeline
├── main.py                     # Application entry point
├── ai_document_enhancer.spec   # PyInstaller build specification
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Testing

### Unit Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

### UI Tests (requires a display or Xvfb on Linux)

```bash
# Windows / macOS (native display)
pytest tests/test_ui.py -v

# Linux (headless)
xvfb-run pytest tests/test_ui.py -v
```

### Mutation Testing

```bash
mutmut run --paths-to-mutate=app/core/order_calculation.py \
           --tests-dir=tests/ \
           --runner="python -m pytest tests/test_mutation.py -x -q"
mutmut results
```

---

## Development Setup

If you want to run the application from source (without building an `.exe`) for development or debugging purposes:

```bash
cd SE_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

This launches the application directly via Python. Any code changes take effect immediately on the next run without needing to rebuild.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `TesseractNotFoundError` | Install Tesseract and add it to PATH, or set the path in **Settings → Tesseract Path** |
| `PDFInfoNotInstalledError` / Poppler missing | Install Poppler and add `bin/` to PATH (see [Prerequisites](#prerequisites)) |
| Grammar stage fails with Java error | Install JRE 8+ (`java -version` to verify). LanguageTool requires Java |
| Models download slowly on first run | DistilBART (~500 MB) and T5-paraphrase (~250 MB) download from Hugging Face on first use. Subsequent runs use the local cache at `~/.cache/huggingface/` |
| Application won't start after PyInstaller build | Run from a terminal (`dist\AIDocumentEnhancer\AIDocumentEnhancer.exe`) to see error output. Common fix: add missing `--hidden-import` entries to the spec file |
| `ImportError: No module named 'sklearn...'` in built exe | Ensure `sklearn.utils._cython_blas` and `sklearn.neighbors._typedefs` are listed in `hiddenimports` (already included in the spec file) |
| High memory usage during summarization | Abstractive summarization loads a transformer model into RAM. Use extractive mode for lower memory consumption, or close other applications |
| OCR quality is poor | Ensure the input image has at least 300 DPI. Enable preprocessing (adaptive threshold + CLAHE) in the pipeline config |

---

## License

This project is developed for educational purposes as part of a Software Engineering course assignment.
