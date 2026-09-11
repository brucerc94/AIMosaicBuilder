# AI Mosaic Builder

AI Mosaic Builder is a local desktop application for analyzing large photo collections, finding useful portrait images, ranking them, enforcing user-defined requirements, reducing visual repetition, and generating an optimized mosaic project.

The application is built with **Python + PySide6** and uses a local multimodal GGUF model through **llama-cpp-python** for image analysis.

The project is designed around one principle: **analysis, selection, layout, preview, and export are separate stages**. This keeps long-running inference out of the UI thread and allows the mosaic layout to be recalculated whenever the user changes the selection rules.

## Workflow

The current workflow is:

```text
Open Folder
      ↓
Discover images
      ↓
Load per-folder cache (if available)
      ↓
Show images immediately in the UI
      ↓
Analyze
      ↓
Load vision model once
      ↓
Analyze uncached images sequentially
      ↓
Detect people / bounding boxes
      ↓
Calculate ranking
      ↓
Apply hard requirements + soft preferences
      ↓
Generate Mosaic
      ↓
Recalculate selection + layout from current user choices
      ↓
Optimize crop + zoom + packing
      ↓
Preview
      ↓
Export JSON or Export Mosaic image
```

`Open Folder` intentionally does **not** load Gemma and does not run inference. It only discovers the images and restores compatible cached analysis so the collection is visible immediately. The expensive model work starts when the user presses **Analyze**.

## Main Features

### Local multimodal analysis

The vision engine loads a local GGUF model and an associated multimodal projector. The repository is prepared for Gemma-4 and dynamically checks the installed `llama-cpp-python` capabilities before selecting the multimodal mechanism.

For Gemma-4, the application uses `Gemma4ChatHandler` when it is available.

The analysis prompt is no longer hardcoded inside Python. It lives in:

```text
prompts/image_analysis.txt
```

The file contains separate `[SYSTEM]` and `[USER]` sections, making the prompt directly editable and version-controlled.

### Structured image analysis

Each analyzed image can contain:

```json
{
  "has_person": true,
  "person_count": 1,
  "main_subject_is_person": true,
  "person_visibility": 0.95,
  "face_visible": true,
  "body_visible": true,
  "occluded": false,
  "blur": 0.05,
  "composition": 0.90,
  "image_quality": 0.92,
  "subject_quality": 0.94,
  "mosaic_value": 0.91,
  "user_request_score": 0.88,
  "visual_tags": {
    "framing": "full_body",
    "orientation": "front",
    "gender_presentation": "unknown",
    "content_rating": "safe",
    "pose": "walking",
    "looking_at_camera": true
  },
  "reject": false,
  "reject_reason": "",
  "notes": "Full-body subject, sharp image, clear separation from the background."
}
```

The model is instructed to use `unknown` whenever an attribute cannot be determined reliably.

Supported visual attributes include:

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

`content_rating` is now normalized around the application-level concepts **SFW** and **NSFW**. Older cached values such as `suggestive` or `explicit` are accepted for backwards compatibility and normalized internally.

### Custom Prompt scoring

The **Filters** tab includes an optional `Custom Prompt` field.

Examples:

```text
only walking
all people jumping
photos on the beach
people hugging
full-body photos at sunset
```

This is not a second search system and it does not create an unlimited tag database. Instead, the custom request is appended to the **same multimodal analysis request** and the model returns one scalar:

```json
"user_request_score": 0.0
```

The value ranges from `0.0` to `1.0`:

```text
0.0 = does not satisfy the request
0.5 = partially satisfies the request
1.0 = satisfies the request very well
```

The custom request has a configurable **Request Weight**. The result is combined with the normal quality score during ranking.

For example:

```text
Base quality score      90
Custom request match    95
Request weight          30%

Final score = 90 × 70% + 95 × 30% = 91.5
```

The custom request is persisted in the application settings. Cache entries are also keyed with a hash of the current request, so results from one request are never silently reused for a different request.

### Person detection

LLM analysis and physical person detection are separate layers.

The multimodal model evaluates the image semantically. A dedicated detector produces actual person bounding boxes used later for crop generation and layout optimization.

The detector backend is replaceable and currently supports a YOLO-based implementation.

### Hard requirements and soft preferences

The **Filters** tab contains two different concepts.

Hard requirements define what the final selection is allowed to contain. Examples include:

```text
Minimum Face-only images
Minimum Full-body images
Minimum Front images
Minimum Side images
Minimum Back images
Minimum Male images
Minimum Female images
Minimum Face-visible images
Minimum Body-visible images
Minimum image quality
Minimum person visibility
Exclude blurry
Exclude heavily occluded
Content policy: Don't care / Safe only / NSFW only
```

Soft preferences do not block an image; they only improve its ranking. Examples include:

```text
Prefer face-only
Prefer full-body
Prefer front
Prefer side
Prefer back
Prefer solo person
Prefer face visible
Prefer body visible
```

A photo can satisfy more than one requirement while still consuming only one mosaic slot.

### Ranking

Every eligible image receives a deterministic score from `0` to `100`.

The current base ranking weights are:

```text
Technical quality   20%
Composition         15%
Person visibility   20%
Subject quality     15%
Sharpness           15%
Face visibility     10%
Mosaic value         5%
```

Additional penalties are applied for conditions such as high blur, occlusion, low subject visibility, poor composition, and very low image quality.

Soft preferences add a small bonus when enabled.

When a `Custom Prompt` exists, its `user_request_score` becomes another ranking component through the configured Request Weight.

After scoring, images are sorted by final score and receive a rank.

### Diversity

The application does not blindly select the top N scores. A perceptual-hash similarity step reduces repeated or nearly identical images so the final mosaic contains more visual variety.

The selection process therefore combines:

```text
Base score
   ↓
Penalties
   ↓
Custom request score (when enabled)
   ↓
Hard eligibility
   ↓
Required composition counts
   ↓
Manual include / exclude decisions
   ↓
Diversity / similarity reduction
   ↓
Final mosaic candidates
```

## Mosaic selection and regeneration

Manual selection changes are first-class inputs to the project.

The user can:

```text
✓ Include in Mosaic
✗ Exclude from Mosaic
```

When a previously selected image is excluded, the application tries to fill the newly opened slot with the next eligible candidate.

More importantly, pressing **Generate Mosaic** after changing manual decisions starts a **fresh selection/layout pass**. The previous generated `SELECTED` state and previous layout are discarded, while manual include/exclude decisions remain authoritative.

This prevents an old layout from contaminating a new one.

Example:

```text
Target = 6

Generate
→ A B C D E F

Exclude C
Include H

Generate again
→ recompute ranking/selection/layout from current rules
→ six-image result using the updated manual choices
```

## Dynamic mosaic layout

`Target Images` supports both:

```text
0 = Automatic
```

and any fixed positive number supported by the UI.

### Automatic mode

AUTO is designed to answer the question:

> How many images can actually fit well on this canvas?

The optimizer considers the configured canvas, detected person size, crop dimensions, minimum subject size, target subject size, padding, zoom, overlap, and available space.

It is not based on a hardcoded number such as 12 or 20.

The target subject settings are configurable in the UI:

```text
Min Subject Size
Target Subject Size
```

They are expressed as a percentage of canvas height and converted into pixel targets for layout evaluation.

The optimizer therefore allows a small face crop to use more zoom while a larger full-body crop can use less zoom, helping the system make better use of the available canvas.

### Fixed mode

When the user specifies a number such as `5`, `6`, or `20`, the optimizer tries to produce exactly that many placements while satisfying the configured layout constraints.

### ImageMosaicView compatibility

The layout logic is designed around the behavior used by the companion `ImageMosaicView` application.

The generated project stores source-image references plus normalized crop coordinates and zoom values rather than producing permanent crop files.

## Preview

After **Generate Mosaic**, the **Preview** button becomes available.

Preview uses the generated selection and renders the mosaic workflow so the user can inspect the result before exporting.

The preview is intentionally separate from analysis and from the final image export.

## Export

There are two final export paths.

### Export JSON

The project export stores the recipe needed to reconstruct the mosaic from the original images, including:

```text
type
filename
crop coordinates
zoom
canvas size
```

Example:

```json
[
  {
    "type": "body",
    "filename": "photo_001.jpg",
    "coords": [0.15, 0.08, 0.61, 0.94],
    "zoom": 0.73,
    "canvas_size": [1080, 960]
  }
]
```

Coordinates are normalized to `0..1`.

### Export Mosaic

**Export Mosaic** renders the actual mosaic image from the generated selection/layout.

Supported output formats are:

```text
PNG
JPEG
WEBP
```

The renderer runs in a background worker so image export does not intentionally block the Qt UI thread.

## Cache

Analysis cache is stored per source folder:

```text
G:/Pictures/MyCollection/
├── photo001.jpg
├── photo002.jpg
└── .aimosaic/
    └── analysis_cache.json
```

The cache is based on the image file hash and the analysis configuration relevant to the model result.

The current cache identity includes the model name and a hash of the `Custom Prompt`, so the following are treated as separate analysis variants:

```text
No custom request
"only walking"
"people hugging"
"photos on the beach"
```

`Open Folder` loads compatible cached analysis immediately so images can be displayed before any model work starts.

`Analyze` then checks each image and skips cache hits. Only uncached or invalidated images go through multimodal inference.

## Model lifetime and performance design

The vision model is loaded once for an analysis run and reused for the entire image batch. The application does not reload the model for every individual photo.

The analysis pipeline is intentionally sequential at the model level so the same loaded multimodal context can process a large collection without repeatedly paying the model-load cost.

The UI uses Qt background workers for long-running operations such as:

```text
Model loading
Image analysis
Person detection / ranking
Mosaic generation
Mosaic image export
```

## llama.cpp configuration

The UI exposes important inference settings instead of hardcoding them in the analysis pipeline.

Current defaults include:

```text
Context       4096 tokens
Max Tokens     576 tokens
n_batch        512
n_ubatch       512
```

Other configurable values include GPU layers, CPU threads, batch threads, and related runtime options.

The integration detects whether parameters such as `n_batch`, `n_ubatch`, `n_threads_batch`, and Flash Attention are supported by the installed `llama-cpp-python` build before passing them to the runtime.

For GPUs that need it, the application can enable `GGML_CUDA_FORCE_MMQ` and disable Flash Attention accordingly.

## Logging

Normal operation uses application-level logs instead of exposing the full internal llama.cpp debug stream.

Typical useful messages look like:

```text
[MODEL] Loaded | 4.70s
[SOURCE] Open Folder | 83 images | cache 60/83
[ANALYZE] 61/83 | photo.jpg | 11.82s
[GENERATE] Layout | 14 images | fill 87.4%
[EXPORT MOSAIC] mosaic.png | 1.31s
```

Detailed inference profiling and low-level diagnostics are intended for debug mode rather than normal use.

The code also records analysis timing, model-call timing, parse timing, prompt/completion token counts, and generation throughput when those metrics are available.

## Settings persistence

Application settings are stored as JSON in:

```text
data/settings.json
```

Persisted values include the selected model paths, source folder, inference parameters, canvas settings, target image count, crop settings, detector settings, ranking configuration, mosaic requirements, and the Custom Prompt plus its Request Weight.

## UI structure

The UI is organized around the real workflow instead of exposing every setting at once.

### Workflow actions

The main action area contains:

```text
Open Folder
Analyze
Stop
Generate Mosaic
Preview
Export Mosaic
```

### Configuration tabs

The settings area is grouped into:

```text
Project
Analysis
Mosaic
Filters
```

The center area shows the image collection, while the right panel displays details for the selected image, including:

```text
Preview
Scores
Detected Attributes
Crop Preview
AI Notes
Mosaic Selection
```

The detail panel also exposes manual Include/Exclude controls.

## Project structure

```text
AIMosaicBuilder/
│
├── main.py
├── requirements.txt
├── README.md
│
├── engine/
│   ├── vision_llm.py          # Local multimodal inference
│   ├── llama_features.py      # llama.cpp capability detection
│   ├── image_analyzer.py      # Discovery, hashing, cache, analysis pipeline
│   ├── ranking.py             # Scores, requirements, preferences, diversity
│   ├── layout_optimizer.py    # AUTO/FIXED selection and layout optimization
│   ├── cache.py               # Analysis cache
│   ├── models.py              # Dataclasses and data models
│   ├── session.py             # Session persistence
│   ├── storage.py             # JSON settings
│   ├── runtime_config.py      # Runtime adapters
│   ├── project_export.py      # JSON project export
│   └── mosaic_image_export.py # Final mosaic image renderer
│
├── vision/
│   ├── person_detector.py     # Person detection / bounding boxes
│   ├── cropper.py             # Crop and thumbnail utilities
│   └── similarity.py           # Similarity helpers
│
├── prompts/
│   └── image_analysis.txt     # Editable vision prompt
│
├── ui/
│   ├── main.py
│   ├── settings.py
│   ├── settings_extended.py
│   ├── image_grid.py
│   ├── image_detail.py
│   ├── preview.py
│   ├── workers.py
│   ├── mosaic_export_worker.py
│   └── styles.py
│
└── test_vision.py             # Multimodal smoke test
```

## Installation

Install the Python dependencies from `requirements.txt`:

```cmd
python -m pip install -r requirements.txt
```

A working local `llama-cpp-python` build with the required multimodal support is required for vision analysis.

## Basic usage

1. Launch AI Mosaic Builder.
2. Press **Open Folder** and select the photo collection.
3. Confirm the image list appears before model inference starts.
4. Select the local GGUF vision model and matching `mmproj` file.
5. Configure the canvas and target image count.
6. Define required composition, quality rules, and preferences in **Filters**.
7. Optionally enter a **Custom Prompt**, such as `only walking`, and set its Request Weight.
8. Press **Analyze**.
9. Review image scores, detections, attributes, and AI Notes.
10. Manually Include or Exclude images when needed.
11. Press **Generate Mosaic** to build a fresh layout from the current rules and manual choices.
12. Press **Preview** to inspect the generated composition.
13. Press **Export JSON** to create the project recipe or **Export Mosaic** to render the final image.

## Example configuration

```text
Canvas:             1920 × 1080
Target Images:      Automatic
Crop Padding:       40 px
Min Subject:        10% of canvas height
Target Subject:     15% of canvas height
Context:            4096 tokens
Max Tokens:         576 tokens
n_batch:            512
n_ubatch:           512

Requirements:
  Face only:        1
  Full body:        2
  Face visible:     3
  Body visible:     3
  Content:          Safe only

Custom Prompt:
  "people walking on the beach"

Request Weight:
  30%
```

## Design principles

AI Mosaic Builder is intentionally modular:

```text
Photo discovery
       ≠
AI analysis
       ≠
Person detection
       ≠
Ranking
       ≠
Selection
       ≠
Layout
       ≠
Preview
       ≠
Export
```

This separation makes it possible to improve the analyzer, ranking model, layout optimizer, or UI without turning every change into a rewrite of the entire application.

## Current status

The project is under active development. The core workflow is implemented around local multimodal analysis, per-folder caching, configurable ranking and requirements, person detection, custom request scoring, dynamic mosaic layout, manual selection overrides, preview, JSON project export, and final mosaic image export.
