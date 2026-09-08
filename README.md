# Stroke Handwriting Recognition

Stroke-based multilingual handwriting recognition system for handwritten input captured from a mouse, touchscreen, or stylus.

The project originally used a hybrid CNN + GRU model. The current architecture is being refactored around **stroke-only recognition**.

```text
Finger / Stylus
      ↓
Stroke Capture
      ↓
Preprocessing
      ↓
Stroke Encoder
      ↓
Recognition
      ↓
Unicode Text
      ↓
Language Correction
      ↓
Clean Text / Handwriting
```

## Goal

Build a mobile application that can recognize and improve handwritten text in:

```text
English
Українська
Русский
Digits
Punctuation
Common symbols
```

Long-term goal:

```text
raw vector handwriting
        ↓
multilingual recognition
        ↓
language-aware correction
        ↓
clean text
        ↓
clean vector handwriting
```

---

## Current Status

Existing MVP:

* [x] Web drawing canvas
* [x] Pointer/stylus stroke collection
* [x] PNG and JSON sample export
* [x] Dataset collection workflow
* [x] Stroke preprocessing
* [x] Image preprocessing
* [x] Data augmentation
* [x] CNN + GRU hybrid experiment
* [x] TensorFlow training
* [x] Model evaluation
* [x] Confusion matrix
* [x] FastAPI backend
* [x] Browser prediction
* [x] Confidence and Top-K output

Current direction:

```text
Hybrid experimental MVP
        ↓
Engineering refactor
        ↓
Stroke-only production architecture
```

---

## Stroke Representation

Raw input may contain:

```text
x
y
time
pressure
stroke boundaries
```

The baseline model should rely on features available across different datasets:

```text
x
y
dx
dy
stroke_start
```

`time` and `pressure` are optional features and may be evaluated in later experiments.

The current fixed-length character representation will eventually be replaced with variable-length sequences.

---

## Data Strategy

The project supports multiple dataset sources through adapters.

```text
dataset source
      ↓
DatasetAdapter
      ↓
StrokeSequence
      ↓
common preprocessing
      ↓
model
```

Planned sources:

```text
Native dataset      → EN / UK / RU
UJI Pen Characters  → English characters, digits, symbols
UNIPEN              → English characters, words, text
IAM-OnDB            → sequence-recognition research benchmark
```

External datasets are not stored directly in the repository.

Each sample should preserve:

```text
label
language
writer_id
session_id
source_dataset
stroke sequence
```

Train/validation/test splitting should be writer-aware.

---

## Target Architecture

```text
src/handwriting/
├── core/
│   ├── config.py
│   ├── logging.py
│   ├── exceptions.py
│   └── types.py
│
├── data/
│   ├── dataset.py
│   ├── manifest.py
│   ├── split.py
│   └── adapters/
│       ├── native.py
│       ├── uji.py
│       ├── unipen.py
│       └── iam_ondb.py
│
├── preprocessing/
│   ├── strokes.py
│   └── features.py
│
├── models/
│   ├── stroke_encoder.py
│   ├── stroke_classifier.py
│   └── model_factory.py
│
├── training/
│   ├── trainer.py
│   ├── experiment.py
│   └── cli.py
│
├── evaluation/
│   ├── metrics.py
│   └── benchmark.py
│
└── inference/
    ├── model_store.py
    ├── service.py
    └── results.py
```

---

## Tech Stack

```text
Python
TensorFlow / Keras
NumPy
scikit-learn
FastAPI
Pydantic
pytest
ruff
mypy
GitHub Actions
Docker
```

Frontend:

```text
HTML
CSS
JavaScript
Canvas API
Pointer Events API
```

---

# Development Roadmap

## Stage 1 — Engineering Foundation

* [x] Add `pytest`
* [x] Add stroke preprocessing tests
* [x] Add augmentation tests
* [x] Add model smoke test
* [x] Add model save/load test
* [x] Create `pyproject.toml`
* [x] Add Ruff
* [x] Add typing
* [x] Add Mypy
* [x] Move code into `src/handwriting`
* [x] Remove `sys.path.append`

**Result:** existing behaviour is protected before refactoring.

---

## Stage 2 — Core Architecture

* [x] Centralize project paths
* [x] Add centralized application configuration
* [x] Audit and extend domain types
* [x] Add domain exceptions
* [x] Add training configuration
* [x] Add structured logging infrastructure
* [x] Replace development print() calls
* [ ] Add request IDs

**Result:** shared application infrastructure.

---

## Stage 3 — Dataset System

* [ ] Split preprocessing into small functions
* [ ] Add input validation
* [ ] Add feature extraction
* [ ] Create `DatasetRepository`
* [ ] Create dataset manifests
* [ ] Add dataset versioning
* [ ] Add preprocessing versioning
* [ ] Add `writer_id`
* [ ] Add `session_id`
* [ ] Add `source_dataset`
* [ ] Add writer-aware splitting
* [ ] Create native dataset adapter
* [ ] Create UJI adapter
* [ ] Integrate UJI Pen Characters
* [ ] Evaluate UNIPEN integration
* [ ] Add UNIPEN adapter
* [ ] Add IAM-OnDB research adapter

**Result:** different stroke datasets use one internal representation.

---

## Stage 4 — Stroke Character Model

* [ ] Create `StrokeEncoder`
* [ ] Create `StrokeClassifier`
* [ ] Keep GRU as baseline
* [ ] Use geometry-based baseline features
* [ ] Add masking/padding support
* [ ] Add model versioning
* [ ] Add parameter count
* [ ] Add model tests
* [ ] Train on native + UJI data
* [ ] Compare writer-independent results

**Result:** clean stroke-only character recognition baseline.

---

## Stage 5 — Production Training

* [ ] Create `Trainer`
* [ ] Create `ExperimentRun`
* [ ] Create training CLI
* [ ] Add callbacks
* [ ] Measure training stages
* [ ] Store training metadata
* [ ] Store Git commit
* [ ] Store evaluation reports
* [ ] Store training plots
* [ ] Create model registry

**Result:** every experiment is reproducible.

---

## Stage 6 — Production Inference

* [ ] Create `ModelStore`
* [ ] Add model caching
* [ ] Create `InferenceService`
* [ ] Create `RecognitionResult`
* [ ] Add Top-K predictions
* [ ] Add inference timings
* [ ] Create `BenchmarkRunner`
* [ ] Measure p50 / p95 latency
* [ ] Measure model size

**Result:** ML inference is independent from the API.

---

## Stage 7 — Backend and Quality

* [ ] Split FastAPI into routers
* [ ] Move Pydantic models into schemas
* [ ] Add `/api/v1`
* [ ] Remove image requirement from prediction
* [ ] Add language selection
* [ ] Add exception handlers
* [ ] Add health endpoints
* [ ] Add API tests
* [ ] Add integration tests
* [ ] Add coverage
* [ ] Add GitHub Actions
* [ ] Add Docker backend

**Result:** stable API for web and mobile applications.

---

## Stage 8 — Multilingual Characters

* [ ] Create versioned Unicode vocabulary
* [ ] Add English alphabet
* [ ] Add Ukrainian alphabet
* [ ] Add Russian alphabet
* [ ] Add digits
* [ ] Add punctuation
* [ ] Add symbols
* [ ] Collect Ukrainian stroke data
* [ ] Collect Russian stroke data
* [ ] Add language metadata
* [ ] Train multilingual character model
* [ ] Analyze Latin/Cyrillic confusion

**Result:** multilingual isolated-character recognition.

---

## Stage 9 — Words and Sentences

* [ ] Remove fixed 100-point limitation
* [ ] Support variable-length sequences
* [ ] Add padding and masking
* [ ] Create sequence encoder
* [ ] Build BiGRU baseline
* [ ] Evaluate Transformer encoder
* [ ] Add CTC decoder
* [ ] Train on words
* [ ] Train on text lines
* [ ] Add CER
* [ ] Add WER
* [ ] Benchmark with IAM-OnDB

**Result:** stroke sequences become complete Unicode text.

---

## Stage 10 — Text Correction

* [ ] Create `CorrectionService`
* [ ] Integrate English dictionary
* [ ] Integrate Ukrainian dictionary
* [ ] Integrate Russian dictionary
* [ ] Add Hunspell support
* [ ] Add edit-distance candidate generation
* [ ] Add frequency-based ranking
* [ ] Integrate `wordfreq`
* [ ] Add contextual ranking
* [ ] Evaluate correction separately from recognition

Correction pipeline:

```text
recognized text
      ↓
dictionary candidates
      ↓
edit distance
      ↓
frequency ranking
      ↓
corrected text
```

**Result:** recognition errors can be corrected without an LLM.

---

## Stage 11 — Mobile MVP

* [ ] Create mobile application
* [ ] Add drawing surface
* [ ] Add stylus support
* [ ] Capture pressure where available
* [ ] Capture stroke sequences
* [ ] Add language selection
* [ ] Connect FastAPI
* [ ] Display predictions
* [ ] Display corrected text
* [ ] Add handwriting-style rendering

**Result:** first usable mobile application.

---

## Stage 12 — Handwriting Beautification

* [ ] Preserve word layout
* [ ] Preserve line layout
* [ ] Render clean glyphs
* [ ] Research stroke decoder
* [ ] Separate content and handwriting style
* [ ] Generate clean vector strokes
* [ ] Add style conditioning
* [ ] Add user-style mode

**Result:** poor handwriting can be reconstructed as clean handwriting.

---

## Stage 13 — On-Device Production

* [ ] Optimize model size
* [ ] Export mobile model
* [ ] Benchmark on mobile hardware
* [ ] Evaluate CPU / GPU / NPU
* [ ] Add offline recognition
* [ ] Add offline correction
* [ ] Add privacy controls
* [ ] Add model update mechanism

**Result:** offline production mobile handwriting recognition.

---

## Current Model Direction

Character baseline:

```text
Stroke Sequence
      ↓
StrokeEncoder
      ↓
GRU
      ↓
Classifier
      ↓
Unicode Character
```

Future sequence model:

```text
Variable-Length Strokes
        ↓
Sequence Encoder
        ↓
CTC
        ↓
Unicode Text
```

---

## Correction Resources

Planned correction layer:

```text
Hunspell
+
wordfreq
+
edit distance / SymSpell
```

Recognition and correction remain independent components.

---

## How to Run

Current legacy MVP:

```powershell
python -m venv .venv

.\.venv\Scripts\python.exe -m pip install -e ".[dev,legacy]"

.\.venv\Scripts\python.exe -m uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd app/frontend
..\..\.venv\Scripts\python.exe -m http.server 5500 --bind 0.0.0.0
```

These commands will change after the package and CLI refactor.

---

## Target API

```text
POST /api/v1/predict
POST /api/v1/samples

GET /api/v1/models/current

GET /health/live
GET /health/ready
```

Prediction should use stroke data directly and should not require a PNG image.

---

## Final Vision

```text
Finger / Stylus
      ↓
Stroke Capture
      ↓
Multilingual Recognition
      ↓
Language Correction
      ↓
Clean Text
      ↓
Clean Vector Handwriting
```
