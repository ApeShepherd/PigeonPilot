# PigeonPilot

![Our Captain](RealPigeonPilotPic.jpg)

Meet Captain Pigeon — the fearless bird who flies this project.

## Research question

> Does **STDP** in a recurrent LIF reservoir improve **path integration**
> compared to a **fixed** (non-plastic) reservoir?

## Trajectory families

| Name | Meaning |
|------|---------|
| `linear` | straight / direct |
| `turning` | heading jumps (turns) |
| `curved` | wave or sweeping arc |

## Difficulty tags (curriculum)

| Tag | Count | Families | Idea |
|-----|-------|----------|------|
| easy | 60 | linear + gentle turning | short, learnable |
| medium | 60 | sharp turning + mild curved | no pure straight lines |
| hard | 30 | sharp turning + arc curved | long memory / arcs |

Total **150** levels, then ~80/20 train/test.

## Repo layout

| Path | Role |
|------|------|
| `pigeonpilot/` | Reusable Python code |
| `PigeonPilot.ipynb` | Single jury notebook |
| `tests/` | Unit tests |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
jupyter notebook PigeonPilot.ipynb
```

In Cursor: `Cmd+Shift+P` → **Jupyter: Restart Kernel and Run All Cells**.
