# PigeonPilot

![Our Captain](RealPigeonPilotPic.jpg)

Meet Captain Pigeon — the fearless bird who flies this project.

## Research question

> Does **STDP** in a recurrent LIF reservoir improve **path integration**
> compared to a **fixed** (non-plastic) reservoir?

## Trajectory families

Display labels live in `STYLE_LABELS` (`pigeonpilot.curriculum`).

| Name | Meaning |
|------|---------|
| `linear` | straight / direct |
| `turning` | heading jumps (turns) |
| `curved` | wave or sweeping arc |

## Difficulty tags (curriculum)

Single source of truth: `DIFFICULTY_SPECS` in `pigeonpilot.curriculum`.  
Render with `format_curriculum_table()` (also used in the notebook).

Default curriculum size: **150** levels (`curriculum_level_count()`), then ~80/20 train/test.
Default curriculum epochs: `DEFAULT_EPOCHS_PER_DIFFICULTY`.

## Repo layout

Quick map for teammates: **domain** = `paths`, **curriculum / train schedule** = `curriculum`, **plots** = `viz`, **demo** = `PigeonPilot.ipynb`.

| Path | Role |
|------|------|
| `pigeonpilot/paths.py` | Domain types, geometry, single-level generation |
| `pigeonpilot/curriculum.py` | Curriculum SSOT, datasets, training schedules |
| `pigeonpilot/viz.py` | Plotting |
| `PigeonPilot.ipynb` | Single jury notebook |
| `tests/` | Unit tests |
| `pyproject.toml` | Package metadata + optional `snn` / `dev` deps |

## Setup

Requires **Python ≥ 3.10**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
jupyter notebook PigeonPilot.ipynb
```

SNN stack (torch / bindsnet / sklearn) when you start encoding/reservoir work:

```bash
pip install -e ".[snn]"
```

In Cursor: `Cmd+Shift+P` → **Jupyter: Restart Kernel and Run All Cells**.
