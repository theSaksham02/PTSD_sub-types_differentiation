# Documentation Index

Guides for understanding, running, and publishing this research pipeline.
Read in order, or jump straight to what you need.

| # | Document | Read when you want to… |
|---|---|---|
| 01 | [What is remaining](01_WHAT_IS_REMAINING.md) | know the current status and blockers |
| 02 | [GPU training guide](02_GPU_TRAINING_GUIDE.md) | actually train on Colab T4 / A100 |
| 03 | [Code walkthrough](03_CODE_WALKTHROUGH.md) | understand what each file does |
| 04 | [Architecture & diagrams](04_ARCHITECTURE.md) | see the pipeline/model visually |
| 05 | [Undergrad researcher guide](05_UNDERGRAD_RESEARCHER_GUIDE.md) | know what else a paper requires |

## Figures

All generated from `generate_diagrams.py` into `figures/`:

- `01_pipeline_overview.png` — Data → Features → Model → Train → Evaluate → Explain
- `02_model_architecture.png` — 4 inputs → late fusion → 2 heads
- `03_feature_schema.png` — the 28-dim feature vector
- `04_gpu_training_flow.png` — Colab training workflow
- `05_research_roadmap.png` — shipped / blockers / high-value / polish

Regenerate with:

```bash
python generate_diagrams.py
```
