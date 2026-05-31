# NaiveBayes (Infer.NET) — Semi-Supervised Bayesian Naive Bayes

A minimal, educational semi-supervised Bernoulli Naive Bayes classifier using Infer.NET for
probabilistic inference.

- Trains from a CSV with binary features and optional labels.
- Supports semi-supervised learning: unlabeled instances have their labels inferred during training.
- Saves a JSON model (Beta posteriors per feature-class, Dirichlet class means).
- Predicts from a saved model without re-running inference.

## Build

Requires .NET SDK 6+. From the repository root:

```bash
dotnet build
```

## Usage

The CLI has two verbs: `train` and `predict`.

### Train

```bash
dotnet run -- train --train <training.csv> [--out-model <model.json>] [--predict-unlabeled <unlabeled_out.csv>] [--verbose]
```

Outputs:
- `model.json` (default) — learned model as JSON.
- `--predict-unlabeled <file>` — per-instance posteriors for unlabeled training rows.

### Predict

```bash
dotnet run -- predict --model <model.json> --input <input.csv> --output <predictions.csv> [--verbose]
```

Output CSV columns: `instance` (0-based row index), `p0`, `p1`, `predicted`.

## CSV format

A header row is required (the loader skips it). Each data row contains binary feature columns
(`0`/`1` or `false`/`true`) followed by a label column (`0`, `1`, or empty for unlabeled).

Example (3 features, mixed labeled and unlabeled rows):

```
f1,f2,f3,label
1,0,1,1
0,1,0,0
1,1,0,
0,0,1,
```

## Model JSON

`model.json` stores learned Beta parameters (alpha/beta per class per feature) and class
probability means. The `PredictCommand` reconstructs Beta point-mass approximations from these
for prediction, which loses the full posterior uncertainty.

## Assumptions and limitations

- Features must be binary (Bernoulli). Continuous or categorical features are not supported.
- Labels are binary (0/1) only.

## Generative story

The full probabilistic model from priors through latent variables to observed data.

### Notation

| Symbol | Meaning |
|--------|---------|
| $N$ | number of instances |
| $F$ | number of features |
| $C = 2$ | number of classes |
| $i \in \{0 \ldots N-1\}$ | instance index |
| $f \in \{0 \ldots F-1\}$ | feature index |
| $c \in \{0, 1\}$ | class index |

### Hyperparameters (fixed)

| Symbol | Default | Meaning |
|--------|---------|---------|
| $\alpha_f$ | 1.0 | Beta prior pseudocount for feature = true |
| $\beta_f$ | 1.0 | Beta prior pseudocount for feature = false |
| $\alpha_c$ | 1.0 | Symmetric Dirichlet concentration for class probabilities |

All defaults are uninformative (uniform).

### Global latent variables

**Class probability vector** — shared across all instances:

$$\boldsymbol{\pi} \sim \mathrm{Dirichlet}([\alpha_c, \alpha_c])$$

With $\alpha_c = 1$ the prior is uniform over the 2-simplex.

**Per-class, per-feature Bernoulli rate** — one independent draw for every $(c, f)$ pair:

$$\theta_{c,f} \sim \mathrm{Beta}(\alpha_f,\, \beta_f) \quad \forall\; c,\, f$$

$\theta_{c,f}$ is the probability that feature $f$ is present when the class is $c$.

### Per-instance latent variable

**Class label** — drawn from the global class distribution:

$$y_i \sim \mathrm{Discrete}(\boldsymbol{\pi}) \quad \forall\; i$$

For labeled instances $y_i$ is constrained to its observed value; for unlabeled instances it
remains a free latent variable inferred jointly with everything else.

### Observed variables

**Binary features** — conditioned on the instance's class label:

$$x_{i,f} \sim \mathrm{Bernoulli}(\theta_{y_i,\, f}) \quad \forall\; i,\, f$$

All feature values are always observed.

### Full joint distribution

$$p(\boldsymbol{\pi},\,\Theta,\,\mathbf{y},\,\mathbf{X})
  = p(\boldsymbol{\pi})
    \prod_{c,f} p(\theta_{c,f})
    \prod_{i=1}^{N} \left[
      p(y_i \mid \boldsymbol{\pi})
      \prod_{f=1}^{F} p(x_{i,f} \mid \theta_{y_i,f})
    \right]$$

For labeled instances the $y_i$ factor collapses to a point mass at the observed label.

### Posterior inference

Infer.NET runs Expectation Propagation and returns:

- **`Beta[C][F]`** — posterior $\mathrm{Beta}(\alpha', \beta')$ for each $\theta_{c,f}$.
- **`Dirichlet`** — posterior over $\boldsymbol{\pi}$.
- **`Discrete[N]`** (optional) — posterior class distribution $p(y_i \mid \mathbf{X})$ for unlabeled instances.

Prediction uses the posterior means of $\boldsymbol{\pi}$ and $\Theta$ in a log-sum Naive Bayes
scoring rule.
