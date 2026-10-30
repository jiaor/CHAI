# CHAI: Cooling Histories via Amortized Inference

CHAI turns apatite fission-track and (U-Th)/He data into thermal histories under a
second, with calibrated uncertainty. A normalizing flow, trained on millions of synthetic
samples, draws the posterior of time-temperature paths directly from the measurements. No
Monte Carlo search runs on your machine. The package also includes exact, differentiable PyTorch
ports of the fission-track annealing model of Ketcham et al. (2007) and RDAAM of Flowers et al. (2009),
for posterior-predictive checks against your data.

CHAI also runs in the browser on your computer:
<https://web.uvic.ca/~rjiao/chai/>.

Installed as `chai-thermo`, imported as `chai`. [README-pypi.md](README-pypi.md) is the short
guide PyPI shows.

## Install

```bash
pip install chai-thermo
```

Python 3.11 or 3.12. Inference runs well on a CPU (GPU is optional). For Linux users, the
default `torch` wheel from PyPI bundles NVIDIA's CUDA libraries (2.5 GB) which is not necessary
for CHAI. If you do not want the CUDA libraries, install the CPU build before CHAI:

```bash
# Linux only
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install chai-thermo
```

## Use

Try it on a synthetic sample whose true history is known:

```python
from chai import api

family = api.load()                           # the shipped family: a 100 Ma and a 250 Ma member, and the chooser
ex = api.example_sample(family, "pulse")      # or "old" (the 250 Ma member answers), or "random"
print(ex.description)
post = family.infer(ex.sample, n=4000, T_present=ex.T_present)   # T_present: the surface temperature, a range
print(post.summary())                         # quantiles of temperature against time, and which member answered
post.plot(truth=ex.truth)                     # the posterior envelope with the true history, matplotlib
```

Your own sample, in a data table's units:

```python
from chai import api

lengths = [14.1, 13.6, 13.9, 12.8, 14.4, 13.2, 11.9, 13.7, 14.0, 12.5, 13.4, 14.6, 13.1, 12.2, 13.8,
           14.2, 13.0, 12.9, 13.5, 14.8, 11.4, 13.3, 13.9, 12.7, 14.3, 13.6, 12.4, 13.8, 14.5, 13.2,
           12.6, 13.9, 14.0, 13.1, 11.8, 13.7, 14.2, 12.9, 13.4, 14.7, 13.0, 12.3, 13.6, 14.1, 12.8,
           13.3, 14.4, 11.6, 13.9, 13.2, 12.7, 14.0, 13.5, 12.1, 14.3, 13.7, 12.9, 13.1, 14.6, 13.4]
sample = api.Sample(
    aft=api.AFT(age=38.0, n_spontaneous=240, lengths=lengths, dpar=2.1),   # lengths c-axis projected, µm,
    grains=[api.Grain(age=24.1, radius=55, eU=18),                          # or add angles=[...] to project them
            api.Grain(age=27.9, radius=78, eU=35)])
family = api.load()
post = family.infer(sample, n=4000, T_present=(10, 20))
print(post.summary())
pred = api.predict(post, sample)              # what the paths predict for the data: ages, the length distribution
fit = api.check_fit(post, sample, family)     # which data the paths do not predict, with the model's noise added
```

Optional arguments of `infer`: `boxes=[api.Box(t=(55, 65), T=(80, 110))]` for constraints the
paths must pass through, `cooling_only=True` for paths that never reheat (by >5 degree),
and `member=100` or `250` to pick the member yourself. Boxes and `cooling_only` are exact
rejection of the member's draws, and the result reports the acceptance rate. `chai.inputs` saves
a sample, its constraints and the settings as one JSON file, the same file the browser page reads
and writes. `help(api)` is the reference: what the members read, what they assume, and every
warning. Lengths given with `angles=` are projected by Ketcham et al. (2007a); `chai.data.projection`
has the parameter sets.

Every answer comes from one member on its own grid, never a mixture of the two. A posterior is
conditional on the prior the family was trained on.

## The shipped family

Two members, one per time span, and a chooser between them:

| | 100 Ma member | 250 Ma member |
|---|---|---|
| grid | 96 uniform nodes | the same 96 to 100 Ma, then 64 to 250 Ma |
| prior | a slow background plus exhumation pulses and reheating episodes, rates to 100 K/Ma (about 90 as the grid resolves them), 0–200 °C | the same, on 250 Ma |
| trained on | 3 M simulated histories, 12 helium grains each, observed afresh every batch | 3 M |
| network | 12 spline couplings behind a conditional Gaussian, 6.8 M parameters | 16 couplings, 13.8 M |

The chooser gives `p(250 Ma | sample)`; `family.infer` answers with the more probable member and
says so. The members take the present-day surface temperature as an input, so `T_present` is
required. [docs/family.md](docs/family.md) has the prior's parameters, the observation model,
the training recipe and the calibration; [docs/architecture.md](docs/architecture.md) draws the
network.

## The forward models

Exact ports, not learned approximations: batched `(B, N)` in, any device, any float dtype,
differentiable with respect to every input.

```python
import torch
from chai.forward.torchimpl.ketcham import aft_forward
from chai.forward.torchimpl.rdaam import rdaam_age

t = torch.tensor([[0.0, 10.0, 40.0, 100.0]])        # Ma, present first, increasing into the past
T = torch.tensor([[15.0, 45.0, 90.0, 120.0]], requires_grad=True)   # °C
out = aft_forward(t, T, torch.tensor([2.0]), n_sub=64)              # age, mean length, the length pdf
out["age"].sum().backward()                                          # exact d(age)/dT
he = rdaam_age(t, T, radius=torch.tensor([60.0]), U=torch.tensor([40.0]), Th=torch.tensor([80.0]))
```

## Citing

Until the paper describing the method appears, cite the software:

> Jiao, R. (2026). CHAI: Cooling Histories via Amortized Inference (version 0.15.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23823454

## Licence

Copyright (C) 2026 Ruohong Jiao. GPL-3.0-or-later; see [LICENSE](LICENSE).
