# DeepQubit — Claude Instructions

## Wiki

Project notes live in `notes/`. The central index is `notes/index.md`; topic pages branch off from there.

**Keep the wiki current.** After any significant code change, update the relevant wiki page(s). A "significant change" includes:
- New or renamed classes/functions in `domains/` or `utils/`
- Changes to gate sets, encodings, or domain parameters
- New or modified scripts in `scripts/`
- Changes to data file formats or directory layout
- New dependencies in `requirements.txt`

Wiki pages to consider updating per area:

| Changed file(s) | Update |
|-----------------|--------|
| `domains/qcircuit.py` | `notes/domain.md` |
| `utils/matrix_utils.py`, `utils/hurwitz.py` | `notes/encodings.md`, `notes/utils.md` |
| `utils/perturb.py` | `notes/utils.md` |
| `scripts/` | `notes/training.md` |
| `data/` structure or formats | `notes/data.md` |
| Domain string parsing | `notes/index.md` (quick reference table) |

Keep notes concise — prefer updating existing content over adding new sections.

## Project

Quantum circuit synthesis using RL and search. See `notes/index.md` for a full overview.

Key entry points:
- Domain: `domains/qcircuit.py`
- Train: `bash scripts/train.sh`
- Solve: `bash scripts/solve.sh`
- Setup (run each shell): `source setup.sh`
