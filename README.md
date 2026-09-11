<div align="center">
  <img src="assets/aimosaicbuilder-icon.svg" width="96" alt="AI Mosaic Builder icon" />
  <h1>AI Mosaic Builder</h1>
  <p><strong>Turn large photo collections into intelligent, customizable portrait mosaics.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Status-In%20Development-orange" alt="Status: In Development" />
    <img src="https://img.shields.io/badge/Version-0.1.0-6D5EF7" alt="Version 0.1.0" />
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/PySide6-Desktop_UI-41CD52?logo=qt&logoColor=white" alt="PySide6" />
    <img src="https://img.shields.io/badge/llama.cpp-Local_Vision-8A2BE2" alt="llama.cpp" />
    <img src="https://img.shields.io/badge/AI-Local--First-111827" alt="Local-first AI" />
  </p>
</div>

> **Development status: In Development.** AI Mosaic Builder is an early-stage `0.1.0` project. The core workflow is functional, but the UI, scoring model, inference integration, cache format, and layout heuristics are still being actively improved and may change between commits.

AI Mosaic Builder is a local desktop application for analyzing large collections of photographs, identifying useful portrait images, ranking them, applying user-defined requirements, and generating optimized mosaic layouts.

It combines **local multimodal vision**, **real person detection**, **quality-aware ranking**, **diversity filtering**, **dynamic crop/zoom**, and **viewer-compatible layout generation** in a single workflow.

> **Local-first by design:** the core image-analysis pipeline runs on your machine using a local GGUF vision model through `llama-cpp-python`. No cloud vision API is required.

## Highlights

| Feature | Description |
|---|---|
| **AI photo analysis** | Evaluates people, visibility, framing, pose, quality, composition, blur, and mosaic value. |
| **Custom Prompt scoring** | Add requests such as `only walking`, `people hugging`, or `photos at the beach`. The vision model scores how well each image matches the request during the same analysis pass. |
| **Smart ranking** | Combines technical quality, subject quality, visibility, preferences, penalties, diversity, and Custom Prompt relevance. |
| **Requirements & filters** | Set minimum counts for face-only, full-body, front, side, back, male/female, face visibility, body visibility, quality, and SFW/NSFW policy. |
| **Diversity control** | Uses perceptual hashing (pHash) to reduce near-duplicate selections. |
| **AUTO layout** | Finds a practical image count, crop, zoom, and packing for the configured canvas. |
| **Manual overrides** | Include or exclude individual photos while preserving the rest of the current analysis. |
| **Persistent cache** | Reuses analysis for unchanged files and separates cache entries by model + Custom Prompt. |
| **Preview** | Review the generated mosaic before exporting it. |
| **Export Mosaic** | Render the selected layout as a full-resolution PNG, JPEG, or WEBP image. |
| **Export JSON** | Export crop coordinates, zoom, source filenames, and canvas metadata for the companion viewer workflow. |

## Workflow

```text
┌──────────────┐
│  Open Folder │  Discover photos + load matching cache
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Configure       │  Canvas, target count, filters,
│ Filters         │  requirements, preferences, Custom Prompt
└──────┬──────────┘
       │
       ▼
┌──────────────┐
│    Analyze   │  Load model → analyze uncached photos →
│              │  detect people → rank → post-process
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Generate Mosaic │  Recalculate selection + crop + zoom + layout
└──────┬──────────┘
       │
       ├──────────────┬───────────────┐
       ▼              ▼               ▼
   Preview       Export JSON     Export Mosaic
```

### Important workflow rules

**Open Folder is cheap.** It discovers images and loads matching cached results. It does not load the vision model or run inference.

**Analyze is the expensive step.** Only uncached images need multimodal inference. Existing cache entries are reused when the file, model, and Custom Prompt match.

**Generate Mosaic is independent from analysis.** Changing canvas size, target image count, padding, or manual selection decisions does not require re-running the full photo analysis.

**Generate Mosaic recalculates from current state.** If a user excludes one selected image or includes another image, the next generation discards the previous generated layout and recomputes the selection and layout from the current analysis plus manual overrides.

## AI Analysis

Each image produces structured analysis such as:

```text
has_person
person_count
main_subject_is_person
person_visibility
face_visible
body_visible
occluded
blur
composition
image_quality
subject_quality
mosaic_value
user_request_score
reject
reject_reason
notes
```

### Visual attributes

```text
framing:
  face_only | head_shoulders | upper_body | half_body | full_body | unknown

orientation:
  front | three_quarter_front | side | three_quarter_back | back | unknown

gender_presentation:
  male | female | unknown

content_rating:
  safe | nsfw | unknown

pose:
  standing | seated | lying | walking | other | unknown

looking_at_camera:
  true | false
```

The analysis prompt instructs the model to use `unknown` whenever a visual attribute cannot be determined reliably.

## Custom Prompt

Filters includes a **Custom Prompt** field. This is evaluated by the vision model during `Analyze`; it is not a keyword search and it does not require an ever-growing list of manual tags.

Example:

```text
photos where everyone is walking
```

The model returns:

```json
"user_request_score": 0.0 .. 1.0
```

The score expresses how well the photograph satisfies the request. When a Custom Prompt exists, its score is incorporated into ranking using the configurable **Request Weight** (30% by default).

Examples of requests that can be evaluated without adding new hard-coded tags:

```text
only walking
people hugging
everyone jumping
photos at the beach
people sitting together on a sofa
people looking at the camera
```

The prompt used by the model is stored outside the Python source code at:

```text
prompts/image_analysis.txt
```

## Ranking

The base ranking uses these default weights:

| Signal | Weight |
|---|---:|
| Technical image quality | 20% |
| Composition | 15% |
| Person visibility | 20% |
| Subject quality | 15% |
| Sharpness | 15% |
| Face visibility | 10% |
| Mosaic value | 5% |

The ranking pipeline is conceptually:

```text
Analysis
  ↓
Base score
  ↓
Penalties
  ↓
Hard eligibility
  ↓
Soft preferences
  ↓
Custom Prompt relevance
  ↓
Diversity filtering
  ↓
Final selection
```

A high-quality image can still rank below a slightly weaker image when the latter is much more relevant to the user's requested subject or scene.

## Requirements & Filters

### Required composition

Minimum counts can be requested for:

- Face only
- Full body
- Front
- Side
- Back
- Male
- Female
- Face visible
- Body visible

These requirements are used by the selection logic, not only displayed by the UI.

### Content policy

The UI supports:

```text
Don't care
Safe only
NSFW only
```

The model-level representation is normalized to:

```text
safe | nsfw | unknown
```

Legacy analysis values such as `suggestive` and `explicit` remain readable and are normalized internally for compatibility.

### Quality controls

Available controls include:

- Minimum image quality
- Minimum person visibility
- Exclude blurry images
- Exclude heavily occluded images
- Minimum source width/height
- Similarity threshold

### Soft preferences

The user can bias selection toward:

- Face-only
- Full-body
- Front
- Side
- Back
- Solo person
- Face visible
- Body visible

Preferences improve ranking without becoming hard requirements.

## Mosaic Layout Optimization

The layout stage is intentionally separated from AI analysis.

For each selected image the optimizer works with the detected person crop and determines an appropriate zoom for the configured canvas. This allows small face crops to receive more zoom while large full-body crops can be reduced so they do not dominate the entire canvas.

### AUTO

`Target Images = 0` means **Automatic**.

AUTO determines how many images can reasonably fit using the current candidates, canvas, padding, subject-size rules, and layout constraints rather than using a hard-coded universal count.

The final result depends on:

- Canvas width and height
- Crop aspect ratios
- Padding
- Minimum subject size
- Target subject size
- Requirements
- Ranking
- Manual Include/Exclude decisions

### Fixed target count

A positive `Target Images` value requests an exact count such as `5`, `12`, or `30`.

The crop and zoom are recalculated for that target every time `Generate Mosaic` is pressed.

## Dynamic Crop & Zoom

Crop padding is configured in the UI.

The project stores crop coordinates and zoom metadata instead of permanently creating crop files for each photo. The viewer can therefore reconstruct the crop from the original source image.

## Manual Include / Exclude

Each image can be manually overridden:

- **Include** — require the image to remain eligible/selected.
- **Exclude** — prevent the image from being selected.

If a selected image is excluded and the target is, for example, `6`, the next valid candidate can fill that open slot. When the user presses `Generate Mosaic`, the old layout is discarded and the selection/layout is rebuilt from the current analysis and manual overrides.

## Cache & Performance

Each source folder has its own cache:

```text
<source-folder>/
├── photos...
└── .aimosaic/
    └── analysis_cache.json
```

The cache uses the file SHA-256 as its primary identity.

The cache model key also includes the Custom Prompt, so different requests are kept separate:

```text
same image + same model + same request
→ cache hit

same image + same model + different request
→ new analysis
```

Failed LLM/runtime analyses are not treated as reusable cache results. This prevents a transient software error from permanently turning an image into a rejected cache entry.

## Recommended Model

The current recommended development configuration is the Gemma-4 E4B multimodal GGUF pair:

```text
Main model:
Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf

Vision projector:
mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf
```

The application detects the model family and uses `Gemma4ChatHandler` when the installed `llama-cpp-python` build provides it.

### Starting inference settings

```text
Context:          4096 tokens
Max Tokens:       576
n_batch:          512
n_ubatch:         512
```

These values are exposed through the application settings rather than being permanently hard-coded into the workflow.

For compatible NVIDIA hardware without tensor cores, the runtime can use an MMQ fallback and disable Flash Attention accordingly.

## Logging

Normal operation keeps low-level `llama.cpp` diagnostics out of the user-facing log and focuses on application-level events such as model loading, cache behavior, analysis progress, and export timing.

Use:

```bash
python main.py --debug
```

to enable detailed diagnostics when troubleshooting inference or layout problems.

## Installation

### Requirements

- Python **3.10+** recommended
- A compatible `llama-cpp-python` build
- A multimodal GGUF model and matching vision projector
- CUDA-enabled `llama-cpp-python` is optional for GPU acceleration

The main Python dependencies are listed in `requirements.txt`:

```text
PySide6
llama-cpp-python
Pillow
numpy
opencv-python
ultralytics
imagehash
```

Install them with:

```bash
python -m pip install -r requirements.txt
```

Run the application with:

```bash
python main.py
```

## Settings

Persistent application settings are stored in:

```text
data/settings.json
```

The settings layer covers the source folder, model paths, context, output token limit, GPU/CPU execution parameters, canvas dimensions, target image count, crop padding, subject-size rules, detector configuration, ranking configuration, mosaic requirements, preferences, and Custom Prompt configuration.

Current defaults include:

```text
Context:              4096
Max Tokens:           576
n_batch:              512
n_ubatch:             512
Custom Prompt Weight: 30%
Target Images:        0 (Automatic)
```

## Export

### Export JSON

The project can export a JSON layout describing original source files, normalized crop coordinates, zoom, and canvas metadata.

Example:

```json
[
  {
    "type": "body",
    "filename": "person.jpg",
    "coords": [0.15, 0.08, 0.61, 0.94],
    "zoom": 0.73,
    "canvas_size": [1080, 960]
  }
]
```

### Export Mosaic

The application can render the selected layout directly to a raster image.

Supported formats:

```text
PNG
JPEG
WEBP
```

Rendering is performed in a background worker and reads the original images directly. Permanent intermediate crop files are not required.

## ImageMosaicView Compatibility

AI Mosaic Builder is designed to work with the companion `ImageMosaicView` workflow:

```text
AI Mosaic Builder
      ↓
Analyze + rank + select
      ↓
Calculate crop + zoom + layout
      ↓
mosaic.json
      ↓
ImageMosaicView
      ↓
Read original photos
      ↓
Reconstruct crops dynamically
      ↓
Render the interactive mosaic
```

The builder and viewer remain separate applications so each can evolve independently while keeping a compatible project format.

## Project Structure

```text
AIMosaicBuilder/
├── main.py
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
│   └── vision_llm.py
├── vision/
│   ├── cropper.py
│   ├── person_detector.py
│   └── similarity.py
└── ui/
    ├── image_detail.py
    ├── image_grid.py
    ├── main.py
    ├── mosaic_export_worker.py
    ├── preview.py
    ├── settings.py
    ├── styles.py
    └── workers.py
```

## Design Principles

### Analyze once, reuse often

Expensive multimodal inference is cached and reused whenever the source image, model, and Custom Prompt match.

### Separate analysis from layout

Changing canvas size, target count, padding, or selection decisions should affect layout generation without forcing a complete vision re-analysis.

### Keep prompts editable

The main analysis prompt lives in `prompts/image_analysis.txt` instead of being embedded in the Python source.

### Keep user intent explicit

Natural-language requests are represented by a `user_request_score` rather than an ever-growing collection of predefined scene tags.

### Fail safely

A runtime/model failure is an `ERROR`, not a semantic photo rejection, and failed analyses are not reused from persistent cache.

### Stay compatible with the viewer

Layout and crop metadata are designed to remain compatible with the companion `ImageMosaicView` workflow.

## Current Development Status

**In Development — `0.1.0`.**

The core pipeline is operational, but this project is not yet considered a stable release. Expect active changes to:

- UI organization and interaction flow
- Ranking and diversity heuristics
- Multimodal inference configuration
- Cache schema and compatibility handling
- Automatic layout optimization
- Export and viewer integration

Feedback, testing, bug reports, and reproducible performance logs are especially useful during this stage.

## License

No open-source license has been declared yet. Until a `LICENSE` file is added, the repository should be treated as **all rights reserved**.
