# Troubleshooting

| Problem | Resolution |
| --- | --- |
| `python: command not found` | Create the environment with `python3`; after activation, use `python` |
| Python is older than 3.9 | Install Python 3.9 or newer and recreate `.venv` with that executable |
| `No module named numpy` or `sklearn` | Activate `.venv`, then run `python -m pip install -r requirements.txt` |
| `data/mcn_games.npz not found` | Run the MCN `generate_data.py` command first |
| Missing `mcn_*.npz` or `mcn_*.pkl` models | Run `train_models.py --data data/mcn_games.npz --game-type mcn` |
| Prediction feature-count error | Use the same `num_agents` and `num_rules` used for training |
| Exact generation is slow | Reduce `num_agents` or `num_games`, or use `--label-method monte_carlo` |
| A Monte Carlo run is slow | Reduce `--monte-carlo-samples` for a trial run |
| A result differs slightly on another machine | Confirm the pinned requirements, Python version, command arguments, seed, and label convention |
| Plots fail on a server | The project selects Matplotlib's headless `Agg` backend; ensure Matplotlib can write its cache directory |
| PowerShell blocks activation | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again |

## Clean reinstall

If the virtual environment contains incompatible packages, remove and recreate
it locally, then install the pinned requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Matplotlib cache warning

If Matplotlib reports that its default cache is not writable, choose a writable
directory for the process:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python train_models.py \
  --data data/mcn_games.npz \
  --game-type mcn
```

## Reporting a reproducibility issue

Include the following information:

```bash
python --version
python -m pip freeze
python -m unittest discover -s tests -v
```

Also include the full generation/training commands, the operating system and
hardware, and the generated `results/mcn_model_metrics.csv` file.
