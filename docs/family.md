# The shipped family: prior, observation model, training and calibration

`api.load()` returns one family, named `"v2.2"` (`api.FAMILY_ASSETS`): a 100 Ma member, a
250 Ma member, and a chooser that says which of the two a sample belongs to. Everything the
three were made from is in `configs/`, and the commands are in the README under *Reproducing
the family*. A posterior from CHAI is conditional on everything on this page.

## 1. The prior over thermal histories

`chai.data.priors.PulsePrior` (`configs/prior_v22.yaml`, `configs/prior_v22_250.yaml`), drawn on
temperature directly. A history is

- a **present-day temperature** drawn uniformly on 0–25 °C, which the members take as an input
  rather than infer, and a **net cooling** over the time span drawn uniformly between zero and
  the distance to 200 °C;
- a **cooling backbone** that spends a share of 5–50% of that net cooling on a steady background
  and the rest on a Poisson number of **exhumation pulses**, each a raised-cosine drop over a
  width of 2–50 Ma (log-uniform), placed so that it completes inside the time span, with a mean
  count of **one per time span** (0 and 1 pulses are the most common, 2 fewer, 3 rare; the pulses
  share the cooling by a Dirichlet draw, so one of them usually dominates);
- **extra recent pulses** at 0.10 per Ma over the last 25 Ma, so that histories that were hot
  until recently are common enough to learn (about 20% of 100 Ma draws and 13% of 250 Ma draws
  have a fission-track age under 25 Ma);
- a Poisson number of **reheating episodes** added on top, each a ramp of 2–25 Ma up by 5–150 K
  (log-uniform), a plateau of 0.5–40 Ma, and a ramp back down, completing inside the time span,
  with a mean count of one per time span;
- temperatures held to **0–200 °C** and cooling and heating rates to **100 K/Ma**: each pulse's
  and ramp's width is floored by the cap, and overlaps that still break it are rejected.

With the 2 Ma floors on pulse width and ramp length, a pulse reaches 100 K/Ma only above a 127 K
drop, and on the 96-node grid (1.05 Ma spacing) the steepest segment of such a pulse reads
60–89 K/Ma. Over 20 000 prior paths the steepest grid segment is 48 / 67 / 90 K/Ma at the 95th
percentile, the 99th and the maximum, so the prior is best described as **rates to about
90 K/Ma as the grid resolves them** (about 54 K/Ma past 100 Ma on the 250 Ma member, whose nodes
there are 2.34 Ma apart). Histories that only cool are the draws with no reheating episode;
`cooling_only=True` selects them by rejection, there is no separate cooling-only member.

| | 100 Ma member | 250 Ma member |
|---|---|---|
| grid | 96 uniform nodes, 1.05 Ma apart | the same 96 to 100 Ma, then 64 to 250 Ma (2.34 Ma apart) |
| pulse and episode rate | 0.01 per Ma (mean count 1) | 0.004 per Ma (mean count 1) |
| recent pulses | 0.10 per Ma over 0–25 Ma | the same |
| solver steps per history | 2 048 | 5 120 |

**Grains.** Each history is dated by 12 helium grains with radius 30–110 µm, U 1–150 ppm, Th/U
0.2–10 (so eU up to about 500 ppm), and one fission-track analysis with Dpar 1.2–3.5 µm.

## 2. The observation model

`configs/observation_v2.yaml`, `chai.data.observe.ObservationModel`. The simulated truths are
observed afresh on every training batch:

- **Fission-track age.** The count of spontaneous tracks is Poisson, `Ns ~ Poisson(506 · age ·
  U · area)`, over 20–50 grains of 1 000–6 000 µm² each, so the age's precision follows the age
  and the uranium content; a 2% zeta and a 3% uranium systematic are shared by the sample.
- **Track lengths.** 40–150 confined lengths, each repeating to 0.20 µm; Dpar from 20 etch pits
  scattering by 0.5 µm.
- **Helium ages.** A 2.5% analytical error and a 12% grain-to-grain dispersion beyond it
  (12.3% in all); radius to 5%, eU to 3%; any subset of the 12 grains.
- **Missing data.** 30% of training samples have no lengths, 20% no fission-track data at all,
  30% no helium, so a sample with less data gets its own, wider posterior.

A measured sample is read the same way: the API computes the count-based age error and the
helium error the model's way, warns where the user's own figure disagrees, and flags inputs
outside the training ranges above.

## 3. The members

Two corpora of **3 M histories** each, drawn with seeds 2120 (100 Ma) and 2270 (250 Ma); the
members are trained on 2.4 M rows of each and validated on held-out rows. Each member is the
network of [architecture.md](architecture.md): a set encoder, 12 (100 Ma) or 16 (250 Ma)
rational-quadratic spline couplings behind a conditional Gaussian, in a latent that is the
path's excess over the present day, Cholesky-whitened against the corpus. Training: 200 000
steps of 4 096 rows (4.7 h and 8.7 h on one GPU), a 1 K noise floor on the target that the API
takes back out of the draws. The shipped weights are the 200 000-step snapshots: 6.8 M parameters
(27 MB) and 13.8 M (56 MB).

**The chooser** (`configs/chooser_v22.yaml`, `chai.models.chooser.MemberChooser`) is the same
encoder with a logit head, trained on balanced batches of both corpora to tell a 250 Ma history
from a 100 Ma one given the data and the present-day temperature; its logit is the log Bayes
factor between the members. On 100 000 held-out rows: log loss 0.380, accuracy 0.799, expected
calibration error 0.002. `family.infer` answers with the more probable member (or the one asked
for) and reports the probability; a sample older than 50 Ma with `p(250 Ma)` between 0.2 and
0.8 is reported as one the two members cannot decide between.

## 4. Calibration

Simulation-based calibration on 50 000 held-out rows per member (`scripts/calibration_study.py`,
1 000 draws per row, the floor removed): per node, the share of truths inside the central 68%
and 95% intervals, on all rows and on strata chosen from the data (complete samples, no lengths,
fission track only, helium only, samples near the time span, long residence then late
exhumation, and rows where the member itself puts a fast segment in play), plus the rank of the
truth for the largest cooling rate and a test that the deep past, where no track survives,
follows the prior's conditional on the recent path.

| | 100 Ma member | 250 Ma member |
|---|---|---|
| all rows, 68% / 95% | within ±0.02 of nominal at every node | 68% within ±0.02; 95% over-covers by up to 0.022 |
| complete samples | 95% over-covers by up to 0.021 | 68% over-covers by up to 0.044 |
| late exhumation (old fission-track age, young helium) | within tolerance | 68% over-covers by up to 0.078 |
| near the time span | 95% over-covers by up to 0.020 | 68% under-covers by up to 0.033 |
| fast segments in play | 68% within tolerance, 95% over-covers by 0.037 | over-covers by about 0.03 |
| largest cooling rate | the head puts paths slightly too smooth: the fast tail is under-stated by about a third | the same |
| deep past | intervals 0.03–0.06 narrower than the prior's conditional warrants | 0.06 narrower |

So the shipped members are conservative in the bulk (where they miss, their intervals are too
wide, not too narrow), under-state the fastest cooling rates, and are a little over-confident in
the deep past, where the posterior is the prior. Answering with one member rather than a mixture
of the two costs 1–3 points of coverage at the worst node, on young samples near 100 Ma.

## 5. What a user should carry over

- Report results with the package version; the prior above is part of the answer.
- Give `T_present` as the range you believe, not a point: the members were trained on
  histories whose present day was anywhere in 0–25 °C and condition on the value you pass.
- Outside the training ranges (ages past the time span, eU above about 500 ppm, radii outside
  30–110 µm, Dpar outside 1.2–3.5 µm) the result is an extrapolation, and the API says so.
- Constraint boxes and `cooling_only` are exact rejection of the member's own draws, so the
  acceptance rate is part of the result; below about 1% the remaining draws say more about the
  data than about the box.
