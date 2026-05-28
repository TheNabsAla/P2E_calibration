# P2E Calibration from Conformal P-Values to E-Values

This repository is to reproduce the experiments of the paper: "Set-Preserving Calibration from Conformal P-Values to E-Values".

The repository contains two folders:

- `e-ca`: experiments for E-value conformal aggregation, comparing several aggregation methods for combining conformal prediction sets from multiple models.
- `e-ccp`: experiments for E-value cross-conformal prediction, comparing standard cross-conformal methods with E-value based variants.

Both folders contain the code, configurations, datasets or dataset loaders, and saved results needed for the experiments.

## Reproducing Results

Install the required Python packages for the relevant folder:


```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
then run `main.py`.

For conformal aggregation:

```bash
cd e-ca
python main.py
```

For cross-conformal prediction:

```bash
cd e-ccp
python main.py
```

The scripts save their output tables in the corresponding results folders.

