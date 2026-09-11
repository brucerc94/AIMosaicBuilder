<div align="center">
  <img src="assets/aimosaicbuilder-icon.svg" width="96" alt="AI Mosaic Builder icon" />
  <h1>AI Mosaic Builder</h1>
  <p><strong>Turn large photo collections into intelligent, customizable portrait mosaics.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/PySide6-Desktop_UI-41CD52?logo=qt&logoColor=white" alt="PySide6" />
    <img src="https://img.shields.io/badge/llama.cpp-Local_Vision-8A2BE2" alt="llama.cpp" />
    <img src="https://img.shields.io/badge/AI-Local--First-111827" alt="Local-first AI" />
    <img src="https://img.shields.io/badge/Version-0.1.0-6D5EF7" alt="Version 0.1.0" />
  </p>
</div>

AI Mosaic Builder is a local desktop application for analyzing large collections of photographs, identifying useful portrait images, ranking them, applying user-defined requirements, and generating an optimized mosaic layout.

It combines **local multimodal vision**, **person detection**, **deterministic ranking**, **diversity filtering**, and **viewer-compatible layout optimization** in one workflow.

> **Local-first by design:** image analysis is performed with a local GGUF vision model through `llama-cpp-python`; no cloud vision API is required by the core pipeline.

## Highlights

| Capability | What it does |
|---|---|
| **AI photo analysis** | Evaluates people, visibility, framing, pose, quality, composition, blur, and mosaic value. |
| **Custom Prompt scoring** | Add a natural-language request such as `only walking`, `people hugging`, or `photos at the beach`; Gemma evaluates every image against it during the same analysis pass. |
| **Smart ranking** | Combines visual quality, subject quality, visibility, preferences, and the custom-request score. |
| **Requirements & filters** | Enforce minimum counts for face-only, full-body, front, side, back, male/female, face visibility, body visibility, plus SFW/NSFW policy. |
| **Diversity control** | Uses perceptual hashing to reduce near-duplicate selections. |
| **AUTO layout** | Finds a practical image count, crop, zoom, and packing for the configured canvas. |
| **Manual overrides** | Include or exclude individual images without losing the rest of the generated selection. |
| **Persistent cache** | Reuses analysis results for unchanged files and distinguishes different Custom Prompts. |
| **Preview** | Opens the generated mosaic layout before export. |
| **Export Mosaic** | Renders the final mosaic at the configured canvas resolution. |
| **Export JSON** | Produces a viewer-compatible project describing original files, crop coordinates, zoom, and canvas size. |

## Workflow

```text
┌──────────────┐
│  Open Folder │  Load images + existing cache
└──────┬───────┘
       │
       ▼
┌────────────────┐
│ Configure      │  Canvas, target images, requirements,
│ Filters        │  preferences, Custom Prompt, thresholds
└──────┬─────────┘
       │
       ▼
┌──────────────┐
│    Analyze   │  Load model → analyze uncached images →
│              │  detect people → rank → post-process
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Generate Mosaic │  Recalculate selection + optimize layout
└──────┬──────────┘
       │
       ├───────────────┐
       ▼               ▼
┌────────────┐   ┌───────────────┐
│   Preview  │   │ Export JSON   │
└────────────┘   └───────────────┘
       │
       ▼
┌────────────────┐
│ Export Mosaic  │  Full-resolution PNG / JPEG / WEBP
└────────────────┘
```

### Important workflow behavior

**Open Folder does not load the vision model or run inference.** It discovers images and hydrates whatever matching analysis is already available in the folder cache. The expensive model work begins only when `Analyze` is pressed.

**Generate Mosaic recalculates the layout from the current user state.** If the user changes manual Include/Exclude decisions after a previous generation, the next Generate starts from the current analysis + manual overrides rather than reusing the previous generated layout.

## AI Analysis

Each analyzed image produces structured information including:

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

The model is instructed to return `unknown` when an attribute cannot be determined reliably.

### Custom Prompt

Filters includes a **Custom Prompt** field. This is not a post-processing text search. It is part of the same multimodal analysis request sent to the vision model.

For example:

```text
Custom Prompt:
photos where everyone is walking
```

The model returns:

```json
"user_request_score": 0.0 .. 1.0
```

where `0` means the image does not satisfy the request and `1` means it satisfies it very well.

The score is then incorporated into ranking when a Custom Prompt is present. The default Custom Prompt weight is **30%** and is user-configurable.

This approach avoids maintaining an unbounded list of scene tags such as `hugging`, `jumping`, `walking`, `beach`, `football`, and so on.

## Ranking

The base ranking combines these signals:

| Signal | Default weight |
|---|---:|
| Technical image quality | 20% |
| Composition | 15% |
| Person visibility | 20% |
| Subject quality | 15% |
| Sharpness | 15% |
| Face visibility | 10% |
| Mosaic value | 5% |

The pipeline then applies penalties, soft preferences, hard eligibility checks, and diversity selection.

When a Custom Prompt is active, `user_request_score` is incorporated as an additional ranking signal instead of replacing the base analysis.

The selection pipeline is conceptually:

```text
Image analysis
     ↓
Base quality score
     ↓
Penalties
     ↓
Hard eligibility
     ↓
User preferences
     ↓
Custom Prompt match
     ↓
Diversity filtering
     ↓
Final selection
     ↓
Layout optimization
```

## Requirements & Filters

The mosaic recipe can express both hard requirements and soft preferences.

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

A single image can satisfy multiple categories, but it occupies only one mosaic slot.

### Content policy

The UI provides:

```text
Don't care
Safe only
NSFW only
```

The model-level representation is now normalized to `safe | nsfw | unknown`; legacy cache values remain readable.

### Quality controls

Available controls include:

- Minimum image quality
- Minimum person visibility
- Exclude blurry images
- Exclude heavily occluded images
- Minimum source width/height
- Similarity threshold

### Soft preferences

Preferences can request a bias toward:

- Face-only
- Full-body
- Front
- Side
- Back
- Solo person
- Face visible
- Body visible

These preferences influence selection without becoming hard requirements.

## Mosaic Layout Optimization

The layout optimizer is designed around the same general behavior expected by `ImageMosaicView`.

For each selected image it works with the real detected person crop and determines an appropriate zoom so that the person remains useful at the configured canvas size.

### AUTO mode

`Target Images = 0` means **Automatic**.

AUTO is intended to answer:

> "Given these candidates, this canvas, this padding, and these subject-size rules, how many images can I place while still producing a useful composition?"

The optimizer evaluates combinations using the available crop geometry and packing constraints rather than assuming a fixed image count.

The result can vary depending on:

- Canvas dimensions
- Number and aspect ratio of selected crops
- Padding
- Minimum subject size
- Target subject size
- Manual Include/Exclude choices
- Requirements and ranking

### Fixed target count

A positive `Target Images` value requests an exact number of images, such as `5`, `12`, or `30`.

The optimizer then recalculates crop and zoom for that exact target.

## Dynamic Crop & Zoom

Crop padding is configurable from the UI rather than hard-coded.

The final layout stores crop coordinates and zoom metadata instead of generating a separate crop image for every source file.

This allows the viewer to reconstruct crops dynamically from the original photographs.

## Manual Selection Overrides

Each analyzed image can be manually:

- **Included** — make it eligible/required for selection.
- **Excluded** — prevent it from being selected.

When the user excludes a currently selected image and regenerates the mosaic, the old generated selection is not reused. The current manual Include/Exclude state becomes the source of truth and the layout is recalculated from scratch.

## Cache & Performance

Analysis cache is stored beside the source collection:

```text
<source-folder>/
├── photo01.jpg
├── photo02.jpg
├── ...
└── .aimosaic/
    └── analysis_cache.json
```

The cache uses the file hash to identify unchanged images.

The analysis model key also incorporates the Custom Prompt. Therefore:

```text
Same image + same model + same Custom Prompt
→ reuse cached result

Same image + same model + different Custom Prompt
→ analyze for the new request
```

This keeps the expensive vision work reusable without confusing results generated for different user requests.

## Model Recommendation

The current recommended development model is:

```text
Main model:
Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf

Vision projector:
mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf
```

This is a **4B-class multimodal GGUF setup** and is a practical balance for local image analysis while remaining substantially lighter than larger vision models.

The application uses the corresponding `Gemma4ChatHandler` when the installed `llama-cpp-python` build provides it.

### Recommended starting inference settings

The current application defaults are:

```text
Context:          4096 tokens
Max Tokens:        576
n_batch:           512
n_ubatch:          512
```

GPU layer count, threads, batch settings, context, and token limits are exposed through the application settings rather than being hard-coded into the workflow.

For NVIDIA GPUs without tensor cores, the runtime can enable an MMQ fallback and disable Flash Attention accordingly.

## Local Vision Stack

The project is built around:

- [Python](https://www.python.org/)
- [PySide6](https://doc.qt.io/qtforpython/)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Pillow](https://python-pillow.org/)
- [OpenCV](https://opencv.org/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [ImageHash](https://github.com/JohannesBuchner/imagehash)

## Installation

### Requirements

- Python **3.10+** is recommended.
- A compatible `llama-cpp-python` build is required for the selected multimodal model.
- For Gemma-4, the installed build must provide `Gemma4ChatHandler`.
- A CUDA-enabled `llama-cpp-python` build can be used for GPU acceleration.

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Start the application

```bash
python main.py
```

For detailed diagnostic logging:

```bash
python main.py --debug
```

Normal mode keeps low-level `llama.cpp` output quiet and shows application-level progress instead. `--debug` enables detailed diagnostics.

## Settings

Application settings are stored in:

```text
 data/settings.json
```

Important persisted settings include:

- Last source folder
- GGUF model path
- Vision projector path
- Context size
- Maximum output tokens
- GPU layers
- CPU threads
- Batch / micro-batch values
- Canvas width and height
- Target image count / AUTO mode
- Crop padding
- Minimum and target subject size
- Person detector settings
- Ranking weights
- Mosaic requirements and preferences
- Custom Prompt
- Custom Prompt weight

The current defaults include:

```text
Context:              4096
Max Tokens:           576
n_batch:              512
n_ubatch:             512
Custom Prompt weight: 30%
```

## Prompt Configuration

The model prompt is intentionally kept outside the Python implementation:

```text
prompts/image_analysis.txt
```

The file contains the editable `[SYSTEM]` and `[USER]` prompt sections used by the vision engine.

This makes prompt iteration possible without editing the inference code.

## Export

### Export JSON

The project can export a JSON layout containing references to the original images plus normalized crop coordinates, zoom, and canvas metadata.

Conceptually:

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

`Export Mosaic` renders the final selected layout at the configured canvas resolution.

Supported output formats:

```text
PNG
JPEG
WEBP
```

Rendering happens in a background worker and reads the original images directly; no permanent intermediate crop files are created.

## ImageMosaicView Compatibility

The JSON is designed to work with the companion `ImageMosaicView` workflow.

```text
AI Mosaic Builder
      ↓
analyze + rank + select
      ↓
optimize crops + zoom + layout metadata
      ↓
mosaic.json
      ↓
ImageMosaicView
      ↓
load original photos
      ↓
reconstruct crops dynamically
      ↓
render / interact with the mosaic
```

The builder and viewer remain separate applications so each can evolve independently.

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

### 1. Analyze once, reuse often

The expensive multimodal analysis is cached so repeated work is minimized.

### 2. Separate analysis from layout

Changing canvas size, target count, padding, or selection decisions should not require re-running the entire vision analysis.

### 3. Keep prompts editable

The primary model prompt lives in a text file and is loaded at runtime.

### 4. Keep user intent explicit

The Custom Prompt is evaluated as a numeric relevance score so arbitrary natural-language requests can influence ranking without creating an ever-growing taxonomy of tags.

### 5. Keep layout deterministic

The selected images are laid out using explicit canvas, crop, padding, subject-size, and zoom rules so that exports are reproducible.

### 6. Keep the UI responsive

Heavy operations such as model inference, post-processing, layout generation, and mosaic rendering run outside the main UI thread.

## Current Status

**Version:** `0.1.0`

The current codebase includes the core desktop workflow, local multimodal analysis, person detection, cache, ranking, requirements/preferences, Custom Prompt scoring, manual overrides, layout optimization, Preview, JSON export, and full-resolution mosaic image export.

The project is still under active development, so interfaces and internal APIs may evolve.

## Troubleshooting

### Gemma-4 handler error

Make sure the installed `llama-cpp-python` version/build provides `Gemma4ChatHandler` and that the selected projector matches the model.

### Analysis is unexpectedly slow

Check the effective context, token, batch, micro-batch, thread, and GPU settings in the UI. Run with `--debug` to expose low-level diagnostics.

### A changed Custom Prompt does not reuse the old analysis

This is intentional. The Custom Prompt participates in the cache key because `user_request_score` depends on the exact request text.

### `Generate Mosaic` changes after manual Include/Exclude

This is also intentional. Manual changes invalidate the previous generated selection; Generate recalculates the selection and layout from the current analysis state.

## License

No license file is currently included in the repository. Add an appropriate `LICENSE` before distributing the project publicly.

---

<div align="center">
  <sub>AI Mosaic Builder — local intelligence for turning many photos into one meaningful picture.</sub>
</div>
