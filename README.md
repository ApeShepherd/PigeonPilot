# PigeonPilot

![Our Captain](RealPigeonPilotPic.jpg)

Meet Captain Pigeon — the fearless bird who flies this project.

## Research question

> Can a recurrent LIF reservoir solve a path-finding problem with a ridge regression readout function?
> Does STDP improve the result?

# TODO: add an explanation of the problem with pictures

## Heading convention (frozen)

Compass / navigation degrees — **not** the mathematical polar angle:

| Degrees | Direction | Axis |
|--------:|-----------|------|
| `0°` | North | `+y` |
| `90°` | East | `+x` |
| `180°` | South | `-y` |
| `270°` | West | `-x` |

Unit step: `(dx, dy) = (sin(heading), cos(heading)) * distance`.  
Default resolution: **36 bins → 10°**. A *bin* is one direction bucket (= one input neuron). Indices are **0-based** (Python): bin `0` = first neuron / “neuron 1” in everyday talk.

## Seeds (reproducibility)

| API | Default seed | Notes |
|-----|-------------:|-------|
| `generate_curriculum_dataset` | `42` | Builds the full easy→extreme dataset |
| `generate_level` | `0` | Always an `int` (never `None`) |
| `split_dataset` | `0` | Stratified train/test shuffle |

Always pass an explicit `seed` in experiments so paths stay debuggable.

## Trajectory families

Display labels live in `STYLE_LABELS` (`pigeonpilot.curriculum`).

| Name | Meaning |
|------|---------|
| `linear` | straight / direct |
| `turning` | heading jumps (turns) |
| `zigzag` | alternating left/right zigzags |
| `curved` | wave or sweeping arc |

## Difficulty tags (curriculum)

Single source of truth: `DIFFICULTY_SPECS` in `pigeonpilot.curriculum`.  
Render with `format_curriculum_table()` (also used in the notebooks).

Monostyle skill ladder (720 levels, then ~80/20 train/test):

| Stage | Style | Skill |
|-------|-------|-------|
| `easy` | linear | heading inversion / warm-up (full 36-bin coverage) |
| `medium` | gentle turning | first real path integration |
| `hard` | gentle zigzag (~50–70°) | visible zigzag / moderate cancellation |
| `expert` | sharp zigzag (~90–120°) | strong cancellation |
| `extreme` | curved **arcs only** | sweeping arcs / long memory |

Default curriculum epochs: `DEFAULT_EPOCHS_PER_DIFFICULTY` (few on `easy`), plus `DEFAULT_MIXED_EPOCHS_AFTER` mixed rehearsal.

## Spike encoding (rate coding)

`pigeonpilot.encoding` turns segments into a spike matrix **before** BindsNET:

- Body-fixed ring: bin `0` = beak (forward); the bird always faces travel direction
- Spikes go to the bin that currently points at **geographic North**  
  (`heading_to_bin(H) = (-compass_bin(H)) mod 36` —  
  North `0°`→bin `0`, East `90°`→`27`, South `180°`→`18`, West `270°`→`9`)
- Constant velocity: firing **duration** ∝ distance (`n_steps = max(1, round(distance / (v · dt)))`)
- Output: `float32` array of shape `(T, 36)` — no Torch/BindsNET dependency
- Demo plots: `plot_body_ring_anatomy`, `plot_level_encoding`, `plot_level_ring_frames`
- Encoding helpers default to `dt=1.0`; viz demos often use `dt=0.25` for readable rasters

## Repo layout

| Path | Role |
|------|------|
| `pigeonpilot/paths.py` | Domain types, geometry, single-level generation |
| `pigeonpilot/encoding.py` | Rate-coding spike trains `(T, n_bins)` |
| `pigeonpilot/curriculum.py` | Curriculum SSOT, datasets, training schedules |
| `pigeonpilot/viz.py` | Plotting |
| `Models.ipynb` | Reservoir A vs B + staged readout jury |
| `DatasetVisualizations.ipynb` | Teaching plots for paths / encoding |
| `tests/` | Unit tests |
| `pyproject.toml` | Package metadata + optional `snn` / `dev` deps |

**Dependency direction (one-way):** `paths` ← `encoding` / `curriculum` / `viz`.  
`paths` never imports the others. Public symbols are also re-exported from `pigeonpilot`.

## Setup

Requires **Python ≥ 3.10**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
jupyter notebook DatasetVisualizations.ipynb
```

SNN stack (torch / bindsnet / sklearn) when you start reservoir work:

```bash
pip install -e ".[snn]"
```

In Cursor: `Cmd+Shift+P` → **Jupyter: Restart Kernel and Run All Cells**.

## Status

Shipped: path geometry (incl. zigzag), monostyle curriculum (720), rate-code encoding, teaching plots, and `Models.ipynb` A/B reservoir + staged Ridge readout.
