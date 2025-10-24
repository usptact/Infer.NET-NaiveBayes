NaiveBayes (Infer.NET) - Semi-Supervised Bayesian Naive Bayes

Overview
--------
This project implements a simple Semi-Supervised Bernoulli Naive Bayes classifier using Infer.NET for probabilistic inference. It supports:

- Training a model from a CSV file containing binary features and (optional) labels.
- Semi-supervised learning: training data may contain unlabeled instances; the model can infer labels for those during training and optionally save posteriors.
- Saving a JSON representation of the learned model (feature Beta posteriors and class Dirichlet means).
- Loading a saved model and predicting unlabeled instances from a CSV without re-running inference.

This is a minimal, educational implementation.

How to build
------------
Requires .NET SDK (6.0/7.0/8.0). From repository root:

```bash
dotnet build
```

Basic usage (CLI)
-----------------
The CLI has two verbs: `train` and `predict`.

Train
~~~~~
Train a model from a CSV and save a JSON model file.

Usage:

```bash
dotnet run -- train --train <training.csv> [--out-model <model.json>] [--predict-unlabeled <unlabeled_out.csv>] [--verbose]
```

Example:

```bash
dotnet run -- train --train train.csv --out-model model.json --predict-unlabeled unlabeled_posteriors.csv --verbose
```

Outputs:
- `model.json` (default) — JSON dump describing feature posteriors and class probabilities.
- If `--predict-unlabeled` is supplied, a CSV with per-instance posteriors for unlabeled training items is written.

Predict
~~~~~~~
Load a saved model and predict for unlabeled instances in a CSV.

Usage:

```bash
dotnet run -- predict --model <model.json> --input <input.csv> --output <predictions.csv> [--verbose]
```

Example:

```bash
dotnet run -- predict --model model.json --input sample.csv --output sample_preds.csv --verbose
```

Output format (`predictions.csv`):
- `instance` — zero-based row index in the input CSV (excluding header)
- `p0` — posterior probability of class 0
- `p1` — posterior probability of class 1
- `predicted` — predicted class (0 or 1)

Preparing CSV data
------------------
DataLoader expects CSV files with a header row (the loader currently skips the first line). Each subsequent row must contain N feature columns followed by one label column. Features must be binary values (either `1`/`0` or `true`/`false`), and labels must be `0` or `1`. For unlabeled instances, leave the label column empty.

Training CSV example (3 features):

```
f1,f2,f3,label
1,0,1,1
0,1,0,0
1,1,0,1
```

Unlabeled sample CSV example (same features, unlabeled rows):

```
f1,f2,f3,label
1,0,1,
0,1,0,
```

Notes:
- The number of feature columns in `sample.csv` (prediction) must match what the model was trained on.
- The loader currently skips the first line; include a header even if it is a placeholder.

Model JSON format
-----------------
`model.json` contains a compact representation of learned Beta parameters (per class, per feature) and class means. The `ModelSerializer` writes a JSON object which the `PredictCommand` can read to construct Beta point-mass approximations and class means used for prediction.

Assumptions and limitations
---------------------------
- Features are modeled as Bernoulli (binary). Continuous or categorical features are not supported.
- Labels are binary (0/1) only.
- The implementation stores Beta "posteriors" for features; when serializing the model the loader reconstructs Beta point-mass distributions from stored alpha/beta (or mean fallback). This loses uncertainty compared to keeping full Infer.NET objects.
