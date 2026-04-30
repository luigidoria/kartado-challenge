# Brumadinho Challenge — Evaluation Rubric

> Engineering perspective: how to read a candidate's submission.

---

## 1. Deliverables

These are the concrete artifacts a submission must contain. Missing any of them is an automatic signal — not necessarily disqualifying, but always worth noting.

### Hard deliverables

| # | Artifact | What it proves |
|---|---|---|
| 1 | Public GitHub repository with meaningful commit history | The candidate actually built it incrementally, not dumped a zip at the end |
| 2 | `README.md` — strategy choice, architecture, reproduction steps, limitations | Can they communicate technical decisions to a non-specialist? |
| 3 | Modular code: `src/` with importable modules, notebooks that orchestrate not implement | Understands separation of concerns |
| 4 | Medallion data architecture (Bronze → Silver → Gold), simulated via Delta tables | Knows data engineering concepts beyond raw pandas |
| 5 | PySpark for metadata/table operations | Can operate in a distributed-compute mental model |
| 6 | MLflow: experiment tracking, metric logging, model registry, `pyfunc` wrapper | Understands ML lifecycle, not just model training |
| 7 | Inference entrypoint: accepts a raw image → returns building count + annotated image | The pipeline closes end-to-end |
| 8 | Performance metrics documented: IoU, Precision, Recall, F1 | Knows how to evaluate, not just train |

### Soft deliverables (differentiators)

- Unit tests for geo utilities, metric implementations, postprocessing logic
- Spatial train/val/test split (not random) — signals awareness of geographic data leakage
- Ethical/contextual framing in README — signals maturity
- `requirements.txt` with pinned versions — signals reproducibility discipline

---

## 2. Quality Parameters

### 2.1 Engineering quality

**Reproducibility**
Does it actually run? Can you `git clone` and reproduce inference in under 15 minutes?
Red flags: hardcoded absolute paths, missing dependencies, data not gitignored, seeds not fixed.

**Code structure**
- Modules do work, notebooks orchestrate. A notebook that contains 300 lines of model training logic is a failure of modularity.
- Functions have type hints and docstrings on public interfaces. This is not pedantry — it signals the candidate writes for others, not just themselves.
- Ruff/Black clean. Linting is hygiene, not style preference.

**Data pipeline integrity**
- Bronze layer is immutable. If the candidate transforms data in bronze, they don't understand the pattern.
- Silver layer has the geo alignment, tiling, and train/val split logic. The split must be spatial, not random — random splits on geographic data leak spatial autocorrelation into validation.
- Gold layer contains model-ready tensors and outputs, not raw inputs.

**MLflow usage**
The difference between a candidate who *used* MLflow and one who *understands* it:
- Checkbox usage: `mlflow.log_metric("iou", val)` once at the end.
- Real usage: autolog + manual metric logging per epoch, artifact logging of sample predictions, model registered with stage (`Staging`/`Production`), `pyfunc` wrapper that makes inference a single call.

### 2.2 ML quality

**Architecture choice and justification**
The choice itself matters less than the reasoning. U-Net + EfficientNet-B0 on a CPU budget is reasonable. SAM zero-shot is a valid shortcut. What's not acceptable: no justification, or a choice that ignores compute constraints (e.g. training Mask R-CNN from scratch on 6 images).

**Handling the core constraint: 6 unlabeled images**
This is the central challenge. Candidates who pretend they have enough labeled data to fine-tune from scratch have missed the problem. The right moves are:
- Pretrained model (INRIA, SpaceNet, xBD weights)
- Weak supervision via Microsoft/Google Building Footprints for the AOI
- Zero-shot with SAM/SAM2
- Transfer learning with minimal fine-tuning

**Spatial filter logic**
The KML polygon is a LineString — it must be closed into a polygon. The images are plain PNGs with no embedded georeferencing. The candidate must either:
(a) establish a pixel-to-geo homography using the red line as a reference, or
(b) work entirely in pixel space using the red line extracted from the image.
Candidates who ignore this and just use the KML coordinates directly against pixel coordinates will produce wrong counts. This is a trap that tests spatial reasoning.

**Metric reporting**
IoU, Precision, Recall, F1 are required. A candidate who only reports accuracy on a highly imbalanced mask (most pixels are background) has not understood the evaluation problem. Per-class breakdown (building vs background) and confusion matrix are differentiators.

### 2.3 Communication quality

- README must tell a story: context → problem → approach → results → limitations.
- Limitations section is mandatory and meaningful. "Model may not generalize" is not a limitation. "Model trained on European building styles may undercount Brazilian informal housing with tin roofs" is a limitation.
- Commit history should be readable. `feat(bronze): ingest pipeline with checksum logging` > `update`.

---

## 3. Possible Approaches for Problem Solving

Ranked by feasibility given the constraint (6 PNG images, no georeference, no ground-truth masks).

### Approach A — Pretrained segmentation + weak labels (recommended)

**Architecture:** U-Net with EfficientNet-B0/B2 encoder via `segmentation_models_pytorch`

**Pipeline:**
1. Download Microsoft Building Footprints or Google Open Buildings for the Brumadinho AOI.
2. Establish pixel-to-geo mapping using the red line in the image as a control reference against the KML coordinates (homography or affine transform).
3. Rasterize footprints to pixel-space binary masks → these are weak labels.
4. Tile each image into 512×512 chips. Spatial split: hold out one geographic block for test.
5. Fine-tune pretrained U-Net (ImageNet encoder, INRIA-pretrained if available) on weak labels.
6. Postprocess: connected components → instance polygons → intersect with impact polygon.

**Strengths:** Generalizes, evaluatable, full pipeline demonstrated.
**Weaknesses:** Georeferencing step is non-trivial with plain PNGs. Weak labels introduce noise.
**Signals:** Strong. Shows geo reasoning, data engineering, ML lifecycle, evaluation discipline.

---

### Approach B — Zero-shot with SAM / SAM2

**Architecture:** Meta's Segment Anything Model, prompted with a grid or automatic mask generation.

**Pipeline:**
1. Run SAM automatic mask generation on each image chip.
2. Filter candidate masks by shape/size heuristics (buildings are roughly rectangular, 20–500px²).
3. Extract red line from image pixels → close into pixel-space polygon.
4. Intersect building centroids with impact polygon.
5. No fine-tuning needed.

**Strengths:** No labeled data required. Fast to prototype. SAM generalizes across domains.
**Weaknesses:** No control over what SAM segments (trees, roads, shadows all get masks). Precision will be low without a classifier on top. Hard to get clean IoU numbers without ground truth. MLflow logging is thin — nothing to track without a training loop.
**Signals:** Medium. Shows awareness of foundation models and pragmatism. Weak on MLflow/pipeline depth.

---

### Approach C — YOLO-seg (YOLOv8/v11)

**Architecture:** YOLOv8-seg or YOLOv11-seg pretrained on aerial building datasets.

**Pipeline:**
1. Use a pretrained YOLOv8-seg checkpoint (e.g., trained on xBD or custom aerial data).
2. Run inference directly on full images or tiles.
3. Filter detections spatially using the pixel-space impact polygon.

**Strengths:** Fast inference. Strong pretrained weights available. Instance segmentation is built in (count comes for free).
**Weaknesses:** Requires a pretrained checkpoint on aerial imagery — not trivial to find for buildings specifically. YOLO's anchor assumptions may not fit satellite scale. Less clean integration with SMP/medallion pattern.
**Signals:** Medium-strong if done well. Shows product pragmatism.

---

### Approach D — Classical CV + contour detection (baseline)

**Architecture:** OpenCV edge detection + morphological operations + contour filtering.

**Pipeline:**
1. Detect buildings via Canny edges, morphological closing, contour area filtering.
2. No deep learning.

**Strengths:** Zero compute, zero dependencies, explainable.
**Weaknesses:** Terrible generalization. Shadows, roads, tree clusters all produce false positives. Not what the challenge asks for.
**Signals:** Weak. Acceptable only as a baseline with explicit acknowledgment of its limits.

---

## Evaluator's mental model

Ask three questions about any submission:

1. **Does the pipeline close?** Raw image in → building count + annotated image out. No gaps.
2. **Did they think about the data, or just apply a model?** The 6-PNG-no-georeference constraint is the real problem. Generic model application without addressing it is a red flag.
3. **Would you trust this in production?** Not "is it perfect" — is it reproducible, documented, tested at the edges, honest about limitations?

A submission that scores 7/10 on ML quality but 9/10 on engineering and communication is more valuable than the inverse.
