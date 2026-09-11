<div align="center">
  <img src="assets/aimosaicbuilder-icon.svg" width="96" alt="AI Mosaic Builder icon" />
  <h1>AI Mosaic Builder</h1>
  <p><strong>Turn large photo collections into intelligent, customizable portrait mosaics.</strong></p>
</div>

> **Development status:** `0.1.0` — In Development.

AI Mosaic Builder is a local Windows desktop application for analyzing large photo collections, ranking useful portrait images, applying user-defined requirements, and generating optimized mosaic layouts.

The application combines a local multimodal GGUF vision model, real person detection, quality-aware ranking, pHash diversity filtering, dynamic crop/zoom, persistent analysis cache, manual include/exclude controls, preview, and mosaic export.

## Highlights

- **AI photo analysis** — people, visibility, framing, pose, quality, composition, blur, and mosaic value.
- **Custom Prompt** — natural-language requests such as `only walking`, `people hugging`, or `photos at the beach` are scored by the vision model during analysis.
- **Requirements** — minimum counts for face-only, full-body, front, side, back, male/female, face-visible, and body-visible images.
- **Quality filters** — minimum image quality/visibility, blur and occlusion controls, source dimensions, and content policy.
- **pHash diversity** — reduces near-duplicate selections.
- **AUTO layout** — determines a practical image count and layout for the configured canvas.
- **Target Subject** — influences layout selection toward the requested subject size.
- **Manual Include/Exclude** — override individual candidate selection without re-running analysis.
- **Persistent cache/session** — unchanged files can be reused across application restarts.
- **Preview / Export** — PNG, JPEG, WEBP and JSON layout export.

## Workflow

```text
Open Folder
    ↓
Discover images + restore cache/session
    ↓
Configure Analysis / Requirements / Mosaic
    ↓
Analyze
    ↓
Gemma vision analysis → person detection → ranking → post-process
    ↓
Generate Mosaic
    ↓
Selection → crop/zoom → layout optimization
    ↓
Preview / Export JSON / Export Mosaic
```

`Open Folder` does not run the vision model. `Analyze` is the expensive operation and only uncached or invalidated images require multimodal inference. `Generate Mosaic` works from the current analyzed state, so changing canvas, target count, padding, or manual selection does not require re-analysis.

## Requirements and Filters

Required composition controls include:

- Face only
- Full body
- Front
- Side
- Back
- Male
- Female
- Face visible
- Body visible

Additional hard eligibility controls include:

- Minimum image quality
- Minimum person visibility
- Exclude blurry images
- Exclude heavily occluded images
- Safe only / NSFW only / Don't care
- Minimum source dimensions
- Similarity threshold

Soft preferences can bias ranking toward face-only, full-body, front, side, back, solo person, face-visible, or body-visible images without turning them into hard exclusions.

The vision model also returns normalized visual attributes such as:

```text
framing:             face_only | head_shoulders | upper_body | half_body | full_body | unknown
orientation:         front | three_quarter_front | side | three_quarter_back | back | unknown
gender_presentation:  male | female | unknown
content_rating:      safe | nsfw | unknown
pose:                standing | seated | lying | walking | other | unknown
looking_at_camera:   true | false
```

## Ranking and Diversity

The ranking pipeline is:

```text
Analysis
  ↓
Base quality score
  ↓
Penalties
  ↓
Hard eligibility
  ↓
Soft preferences
  ↓
Custom Prompt relevance
  ↓
pHash diversity filtering
  ↓
Final selection
```

The **Similarity** setting is a pHash Hamming-distance threshold. Lower distance means images are more perceptually similar. It is not a percentage and it is not a pose detector.

The application also computes pHash when a cached/session record reaches selection without one, preventing cached records from bypassing the diversity check.

## Mosaic Layout

`Target Images = 0` means **Automatic**. AUTO considers the available candidates, canvas dimensions, padding, subject-size rules, requirements, ranking, and manual overrides.

A positive `Target Images` value requests an exact target count when enough eligible candidates exist.

The optimizer calculates crop coordinates and zoom metadata from the detected person rather than permanently creating crop files.

### Subject size

- **Min Subject** is a hard lower bound.
- **Target Subject** is a preferred subject size used by layout scoring.

## Cache and Session

Each source folder stores application state under:

```text
<source-folder>/.aimosaic/
├── analysis_cache.json
└── session.json
```

The cache uses the file SHA-256 as its primary identity and separates analyses by model and Custom Prompt. Failed runtime/model analyses are not treated as reusable successful results.

## Recommended Model

The current development configuration uses a Gemma-4 E4B multimodal GGUF pair:

```text
Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf
```

The application detects the Gemma family and uses `Gemma4ChatHandler` when supported by the installed `llama-cpp-python` build.

Recommended starting inference settings:

```text
Context:    4096
Max Tokens: 576
n_batch:    512
n_ubatch:   512
```

## Installation — Windows

### Recommended: one-click installer

Install **Python 3.12**, clone/download this repository, then run:

```text
install_windows.bat
```

The installer creates an isolated `.venv`, installs the normal Python dependencies, asks whether llama.cpp should use CPU or NVIDIA CUDA, installs the corresponding `llama-cpp-python` runtime, and verifies the application imports.

Supported installer choices currently include:

```text
CPU
NVIDIA CUDA 11.8
NVIDIA CUDA 12.1
NVIDIA CUDA 12.4
NVIDIA CUDA 13.0
NVIDIA CUDA 13.2
```

For the complete Windows installation procedure, troubleshooting, and manual installation commands, see:

```text
INSTALL_WINDOWS.md
```

### Why llama.cpp is installed separately

`llama-cpp-python` is intentionally **not** installed by the normal `requirements.txt` file. CPU and CUDA builds are hardware/platform dependent. Keeping it separate prevents a normal dependency installation from silently replacing a working GPU build with a CPU build or attempting an unsuitable local compilation.

The dependency architecture is therefore:

```text
requirements.txt
    ↓
Common Python dependencies
    ↓
install_windows.bat
    ↓
CPU/CUDA llama.cpp runtime
    ↓
AI Mosaic Builder
```

The upstream llama-cpp-python project documents both pre-built CPU/CUDA wheels and source builds with `GGML_CUDA=on`.

### Manual installation

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

CPU llama.cpp:

```bat
python -m pip install "llama-cpp-python>=0.3.33,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

NVIDIA CUDA example:

```bat
python -m pip install "llama-cpp-python>=0.3.33,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Run:

```bat
.venv\Scripts\python.exe main.py
```

Debug mode:

```bat
.venv\Scripts\python.exe main.py --debug
```

## Model Configuration

The model is not downloaded automatically. In **Project → Vision Model**, select:

```text
Model:  Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
mmproj: mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf
```

The model and projector must be a compatible pair.

## Settings Persistence

Application settings are stored in:

```text
data/settings.json
```

The settings include model paths, inference parameters, detector configuration, canvas dimensions, target count, padding, subject-size rules, requirements, ranking preferences, Custom Prompt configuration, and other workflow state.

## Export

### JSON

The exported layout contains source filenames, normalized crop coordinates, zoom, and canvas metadata so the companion viewer can reconstruct the mosaic from the original images.

### Raster Mosaic

Supported output formats:

```text
PNG
JPEG
WEBP
```

Rendering is performed in a background worker and reads the original source images directly.

## Project Structure

```text
AIMosaicBuilder/
├── main.py
├── install_windows.bat
├── INSTALL_WINDOWS.md
├── requirements.txt
├── prompts/
│   └── image_analysis.txt
├── assets/
│   └── aimosaicbuilder-icon.svg
├── engine/
│   ├── cache.py
│   ├── image_analyzer.py
│   ├── layout_optimizer.py
│   ├── llama_features.py
│   ├── models.py
│   ├── mosaic_image_export.py
│   ├── project_export.py
│   ├── ranking.py
│   ├── runtime_config.py
│   ├── session.py
│   ├── storage.py
│   ├── target_subject.py
│   └── vision_llm.py
├── vision/
│   ├── cropper.py
│   ├── person_detector.py
│   └── similarity.py
└── ui/
    ├── exclude_folders.py
    ├── image_detail.py
    ├── image_grid.py
    ├── main.py
    ├── mosaic_export_worker.py
    ├── preview.py
    ├── reanalysis.py
    ├── settings.py
    ├── settings_extended.py
    ├── styles.py
    └── workers.py
```

## Design Principles

### Analyze once, reuse often

Expensive multimodal inference is cached and reused whenever the source image, model, and Custom Prompt match.

### Separate analysis from layout

Changing canvas size, target count, padding, or selection decisions should affect layout generation without forcing a complete vision re-analysis.

### Keep prompts editable

The main analysis prompt lives in `prompts/image_analysis.txt` rather than being embedded in Python source.

### Keep user intent explicit

Natural-language requests are represented by a `user_request_score` instead of an ever-growing collection of hard-coded scene tags.

### Fail safely

A runtime/model failure is an error, not a semantic photo rejection, and failed analyses are not reused from persistent cache.

### Stay compatible with the viewer

Layout and crop metadata are designed for the companion `ImageMosaicView` workflow.

## Current Development Status

**In Development — `0.1.0`.**

The core workflow is operational, but UI organization, ranking/diversity heuristics, multimodal inference configuration, cache/session behavior, and layout heuristics may continue to change between commits.
