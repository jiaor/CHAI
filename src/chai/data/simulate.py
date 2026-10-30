"""Build the simulation corpus.

One corpus, keyed on the **thermal history**.  PLAN.md originally asked for a
10^6-sample AFT corpus and a separate 10^5-sample RDAAM corpus, which was sized
under cost assumptions that `docs/node-count-study.md` later corrected.  Two
corpora drawn independently could never be joined: the Stage 5 sample is
"1 AFT age + length histogram + Dpar + N AHe grains", all from *one* history, and
that object cannot be assembled from two files that never shared a path.  So a
row here is one history, its AFT observation, and ``n_grains`` AHe grains; the
single-observable corpora are views on it.  Measured, that costs 50 minutes
for 2x10^6 histories with 4 grains on 112 cores.

What the file guarantees
------------------------
**Reproducibility.**  Every block of histories is drawn from its own child of one
:class:`numpy.random.SeedSequence`, so a block's contents depend on the corpus
seed and the block index and on nothing else -- not on worker count, not on
completion order, not on whether the run was resumed.  :func:`reproduce` redraws
any single sample and is tested against the stored one.

**Resumability.**  The file is allocated in full up front and carries a
``block_done`` flag per block.  An interrupted run leaves a valid file; rerunning
the same command fills in the blocks that are missing, and produces byte-identical
results to an uninterrupted run.

**Splits that cannot leak.**  ``split`` is assigned once, at creation, over a
shuffled index, and written before any forward model runs.  PLAN.md Sec 6 logs two
inherited failures this closes: the prototype's Keras ``validation_split`` put
short paths in train and long ones in validation exclusively, and its accuracy
plots were computed on ``X_path[:k]``, which lies inside the training split.
Neither is expressible against a corpus that ships its own split column.

**Provenance.**  Git SHA and dirty flag, the prior and grain configs as JSON, the
grid, the AFT calibration, the RDAAM precision, and a SHA-256 of each compiled
oracle, all in the root attributes.

Parallelism is by **process**, never thread: RDAAM mutates file-scope C++ state
(PLAN.md Sec 6.7).  The parent process is the only writer.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from chai.data.priors import GrainPrior, ThermalHistoryPrior, prior_from_config
from chai.forward.native import (
    KETCHAM_2007,
    LENGTH_BINS,
    PREC_BEST,
    calculate_aft,
    calculate_rdaam,
    eU,
)
from chai.paths import fixed_grid

__all__ = ["CorpusSpec", "extend_grains", "provenance", "reproduce",
           "reproduce_extra_grains", "simulate"]

SPLIT_NAMES = ("train", "val", "test")

#: Second word of the split RNG's seed, so the split stream can never collide
#: with the draw stream even though both are derived from ``spec.seed``.
SPLIT_STREAM = 991

#: Last word of the grain-extension RNG's seed.  :func:`extend_grains` appends
#: grains to a corpus whose histories are already drawn and must not move, so its
#: stream has to be provably disjoint from the one that drew them -- see
#: :func:`_extend_seed_sequence`.
EXTEND_STREAM = 773

#: The shipped prior's per-draw latent parameters, stored so the Goal 2 network
#: can be conditioned on them rather than retrained per prior (PLAN.md Sec 5).
#: A prior states its own through ``hyper_names`` -- an episodic prior reports an
#: episode count where this one reports a correlation length -- and the writer
#: asks it; this is the fallback for a file written before that was so.
HYPER_NAMES = ("tau", "rate_sd", "drift", "T_0")

#: Per-history diagnostics carried into the corpus.  Enough to stratify an audit
#: or a training curriculum by what the history actually did, without redrawing.
CARRIED_DIAGNOSTICS = (
    "max_cooling_rate",
    "max_heating_rate",
    "total_variation",
    "reheat_amplitude",
    "monotone",
)


@dataclass(frozen=True)
class CorpusSpec:
    """Everything about a corpus that is not the prior itself.

    Args:
        n_histories: rows in the corpus.
        n_grains: AHe grains simulated per history.  They share the history and
            differ in radius / U / Th, which is exactly the multi-grain aliquot
            Stage 5 needs; one grain would make that impossible to assemble later.
        n_grid: nodes in the stored path vector.  64 is settled in
            ``docs/prior.md``; it is the Goal 2 latent dimension.
        t_max: time span, Ma.  Must match the prior's.
        spacing: node placement.  Uniform is settled in ``docs/prior.md``.
        block_size: histories per worker task.  The prior draw is vectorised over
            a block and is the dominant cost (20 ms/history unbatched against the
            AFT oracle's 0.21), so blocks want to be large enough to amortise it
            and small enough to keep the workers fed and resumption granular.
        seed: the corpus seed.  With this file's configs it determines everything.
        split_fractions: train / val / test.
    """

    n_histories: int
    n_grains: int = 4
    n_grid: int = 64
    t_max: float = 100.0
    spacing: str = "uniform"
    block_size: int = 512
    seed: int = 0
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __post_init__(self) -> None:
        if self.n_histories < 1:
            raise ValueError(f"n_histories must be >= 1, got {self.n_histories}")
        if self.n_grains < 1:
            raise ValueError(f"n_grains must be >= 1, got {self.n_grains}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if not np.isclose(sum(self.split_fractions), 1.0):
            raise ValueError(f"split_fractions must sum to 1, got {self.split_fractions}")
        if any(f < 0 for f in self.split_fractions):
            raise ValueError(f"split_fractions must be non-negative, got {self.split_fractions}")

    @property
    def n_blocks(self) -> int:
        return -(-self.n_histories // self.block_size)

    def block_slice(self, b: int) -> slice:
        lo = b * self.block_size
        return slice(lo, min(lo + self.block_size, self.n_histories))


def _git(*args: str) -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _oracle_digests() -> dict[str, str]:
    """SHA-256 of each compiled oracle, so a corpus names the binary that made it.

    The oracles are rebuilt from source by ``make native`` and are not in git, so
    a git SHA alone does not pin them -- a stale ``native/build`` would otherwise
    be invisible in the provenance.
    """
    build = Path(__file__).resolve().parents[3] / "native" / "build"
    out = {}
    for so in sorted(build.glob("*.so")):
        out[so.name] = hashlib.sha256(so.read_bytes()).hexdigest()
    return out


def provenance() -> dict[str, Any]:
    """Everything needed to say where a corpus came from."""
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "oracles": _oracle_digests(),
        "aft_model": "KETCHAM_2007",
        "rdaam_precision": "PREC_BEST",
    }


def _assign_splits(spec: CorpusSpec) -> np.ndarray:
    """Split labels over a shuffled index, fixed before any sample is simulated.

    Shuffled, because the inherited failure was a split taken over a corpus built
    in blocks of increasing path length.  Fixed at creation, because a split
    decided later is a split that can be decided again.
    """
    rng = np.random.default_rng([spec.seed, SPLIT_STREAM])
    order = rng.permutation(spec.n_histories)
    edges = np.cumsum(np.asarray(spec.split_fractions) * spec.n_histories).astype(int)
    split = np.empty(spec.n_histories, dtype=np.uint8)
    split[order[: edges[0]]] = 0
    split[order[edges[0] : edges[1]]] = 1
    split[order[edges[1] :]] = 2
    return split


def _seed_sequence(spec: CorpusSpec) -> np.random.SeedSequence:
    return np.random.SeedSequence([spec.seed, spec.n_histories, spec.n_grains])


def draw_block(
    prior: ThermalHistoryPrior,
    grain_prior: GrainPrior,
    t_grid: np.ndarray,
    n: int,
    n_grains: int,
    seed_seq: np.random.SeedSequence,
) -> dict[str, np.ndarray]:
    """The latent half of a block: histories and grains, no forward model.

    Split out from :func:`_simulate_block` so :func:`reproduce` can redraw exactly
    what a worker drew without paying for the oracles.
    """
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    T, batch = prior.sample_on(n, rng, t_grid)
    grains = grain_prior.sample(n * n_grains, rng)
    out = {"T": T, **{f"hyper_{k}": v for k, v in batch.hyper.items()}}
    diag = batch.diagnostics()
    out.update({f"diag_{k}": np.asarray(diag[k]) for k in CARRIED_DIAGNOSTICS})
    out.update({f"grain_{k}": v.reshape(n, n_grains) for k, v in grains.items()})
    return out


def _simulate_block(args) -> tuple[int, dict[str, np.ndarray]]:
    """One block: draw it, then run both oracles over it.  Runs in a worker."""
    b, n, n_grains, prior_kind, prior_cfg, grain_cfg, t_grid, seq = args
    prior = prior_from_config(prior_cfg, prior_kind)
    grain_prior = GrainPrior.from_config(grain_cfg)
    out = draw_block(prior, grain_prior, t_grid, n, n_grains, seq)

    aft = {
        "aft_age": np.empty(n, np.float32),
        "aft_oldest_age": np.empty(n, np.float32),
        "aft_mean_length": np.empty(n, np.float32),
        "aft_length_pdf": np.empty((n, len(LENGTH_BINS)), np.float16),
    }
    he = {k: np.empty((n, n_grains), np.float32) for k in ("he_age", "he_corrected_age")}

    # The AFT kinetic parameter is a property of the grain population, so the
    # history's Dpar is the first grain's -- one aliquot, one etch-pit figure.
    dpar = out["grain_dpar"][:, 0]
    for i in range(n):
        a = calculate_aft(t_grid, out["T"][i], kinetic_par=float(dpar[i]),
                          annealing_model=KETCHAM_2007)
        aft["aft_age"][i] = a.age
        aft["aft_oldest_age"][i] = a.oldest_age
        aft["aft_mean_length"][i] = a.mean_length
        total = a.length_pdf.sum()
        aft["aft_length_pdf"][i] = (a.length_pdf / total if total > 0 else a.length_pdf)
        for g in range(n_grains):
            r = calculate_rdaam(t_grid, out["T"][i], float(out["grain_radius"][i, g]),
                                float(out["grain_U"][i, g]), float(out["grain_Th"][i, g]),
                                precision=PREC_BEST)
            he["he_age"][i, g] = r.age
            he["he_corrected_age"][i, g] = r.corrected_age

    out.update(aft)
    out.update(he)
    out["grain_eU"] = eU(out["grain_U"], out["grain_Th"])
    # Anomalies are flagged, never dropped: silently deleting rows would edit the
    # prior, and PLAN.md Sec 6.12 is a case of exactly this kind of artifact
    # (RDAAM returning ~3.5e-5 Ma) that must stay visible in the audit.
    out["valid"] = (
        np.isfinite(out["aft_age"]) & (out["aft_age"] > 0)
        & np.isfinite(he["he_age"]).all(axis=1) & (he["he_age"] > 0).all(axis=1)
        & np.isfinite(he["he_corrected_age"]).all(axis=1)
    )
    return b, out


def _create(path: Path, spec: CorpusSpec, prior, grain_prior, t_grid, compression) -> None:
    """Allocate the file in full, and write everything that does not need an oracle."""
    n, k, g = spec.n_histories, spec.n_grains, spec.n_grid
    chunk = min(spec.block_size, n)
    with h5py.File(path, "w") as f:
        f.attrs.update({k_: json.dumps(v) if isinstance(v, dict) else v
                        for k_, v in provenance().items()})
        f.attrs["spec"] = json.dumps(asdict(spec))
        f.attrs["prior_class"] = type(prior).__name__
        f.attrs["prior_config"] = json.dumps(prior.to_config())
        f.attrs["grain_config"] = json.dumps(grain_prior.to_config())
        f.attrs["seed_entropy"] = json.dumps(_seed_sequence(spec).entropy)
        f.attrs["split_names"] = list(SPLIT_NAMES)
        f.create_dataset("t_grid", data=t_grid)
        f.create_dataset("length_bins", data=np.asarray(LENGTH_BINS))

        def make(name, shape, dtype, chunks):
            f.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks,
                             compression=compression)

        make("T", (n, g), np.float32, (chunk, g))
        make("aft/length_pdf", (n, len(LENGTH_BINS)), np.float16, (chunk, len(LENGTH_BINS)))
        for name in ("aft/age", "aft/oldest_age", "aft/mean_length", "aft/dpar"):
            make(name, (n,), np.float32, (chunk,))
        for name in ("he/age", "he/corrected_age", "he/radius", "he/U", "he/Th", "he/eU"):
            make(name, (n, k), np.float32, (chunk, k))
        f.attrs["hyper_names"] = list(prior.hyper_names)
        for name in prior.hyper_names:
            make(f"hyper/{name}", (n,), np.float32, (chunk,))
        for name in CARRIED_DIAGNOSTICS:
            make(f"diag/{name}", (n,), np.float32, (chunk,))
        make("valid", (n,), bool, (chunk,))
        f.create_dataset("split", data=_assign_splits(spec))
        f.create_dataset("block", data=np.repeat(np.arange(spec.n_blocks), spec.block_size)
                         [:n].astype(np.int32))
        f.create_dataset("block_done", data=np.zeros(spec.n_blocks, dtype=bool))


def _store(f: h5py.File, sl: slice, out: dict[str, np.ndarray]) -> None:
    f["T"][sl] = out["T"]
    f["aft/length_pdf"][sl] = out["aft_length_pdf"]
    for name in ("age", "oldest_age", "mean_length"):
        f[f"aft/{name}"][sl] = out[f"aft_{name}"]
    f["aft/dpar"][sl] = out["grain_dpar"][:, 0]
    for name in ("age", "corrected_age", "radius", "U", "Th", "eU"):
        key = f"he_{name}" if name in ("age", "corrected_age") else f"grain_{name}"
        f[f"he/{name}"][sl] = out[key]
    for name in f.attrs.get("hyper_names", HYPER_NAMES):
        f[f"hyper/{name}"][sl] = out[f"hyper_{name}"]
    for name in CARRIED_DIAGNOSTICS:
        f[f"diag/{name}"][sl] = out[f"diag_{name}"]
    f["valid"][sl] = out["valid"]


def simulate(
    path: str | Path,
    spec: CorpusSpec,
    prior: ThermalHistoryPrior | None = None,
    grain_prior: GrainPrior | None = None,
    *,
    workers: int | None = None,
    compression: str | None = "lzf",
    progress: bool = True,
) -> Path:
    """Generate, or finish generating, a corpus at ``path``.

    Rerunning the same call on an existing file resumes it: blocks already marked
    done are skipped, and the result is identical to an uninterrupted run.
    """
    import multiprocessing as mp

    path = Path(path)
    prior = prior or ThermalHistoryPrior(t_max=spec.t_max)
    grain_prior = grain_prior or GrainPrior()
    if prior.t_max != spec.t_max:
        raise ValueError(f"prior time span {prior.t_max} != spec time span {spec.t_max}")
    t_grid = fixed_grid(spec.n_grid, spec.t_max, spec.spacing).numpy()

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _create(path, spec, prior, grain_prior, t_grid, compression)

    with h5py.File(path, "r") as f:
        # Round-trip both sides: JSON has no tuple, so a direct compare against
        # asdict() would report every existing file as a mismatch.
        if json.loads(f.attrs["spec"]) != json.loads(json.dumps(asdict(spec))):
            raise ValueError(f"{path} was written for a different CorpusSpec")
        todo = np.flatnonzero(~f["block_done"][:]).tolist()
    if not todo:
        return path

    children = _seed_sequence(spec).spawn(spec.n_blocks)
    # The child SeedSequence travels whole.  Sending ``child.entropy`` instead
    # would be silently wrong: spawned children share their parent's entropy and
    # differ only in ``spawn_key``, so every block would draw the same histories.
    jobs = [(b, spec.block_slice(b).stop - spec.block_slice(b).start, spec.n_grains,
             type(prior).__name__, prior.to_config(), grain_prior.to_config(),
             t_grid, children[b])
            for b in todo]

    t0 = time.perf_counter()
    workers = workers or mp.cpu_count()
    with mp.Pool(workers) as pool, h5py.File(path, "r+") as f:
        for done, (b, out) in enumerate(
                pool.imap_unordered(_simulate_block, jobs, chunksize=1), start=1):
            _store(f, spec.block_slice(b), out)
            f["block_done"][b] = True
            f.flush()
            if progress and (done % 10 == 0 or done == len(jobs)):
                rate = done / (time.perf_counter() - t0)
                print(f"\r{done}/{len(jobs)} blocks · {rate * spec.block_size:.0f} "
                      f"histories/s · {(len(jobs) - done) / rate / 60:.1f} min left",
                      end="", flush=True)
    if progress:
        print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min")
    return path


def reproduce(path: str | Path, index: int) -> dict[str, np.ndarray]:
    """Redraw sample ``index`` from the corpus's own provenance.

    The corpus stores a block index rather than a per-sample seed, because the
    prior draw is vectorised over a block and drawing one history at a time costs
    about twice as much per history.  Reproducing one sample therefore redraws its
    block -- a few seconds -- and this returns that sample's row of it.  The point
    is that nothing outside the file is needed to do so.
    """
    with h5py.File(path, "r") as f:
        spec = CorpusSpec(**{k: tuple(v) if k == "split_fractions" else v
                             for k, v in json.loads(f.attrs["spec"]).items()})
        prior = prior_from_config(json.loads(f.attrs["prior_config"]),
                                  f.attrs.get("prior_class", "ThermalHistoryPrior"))
        grain_prior = GrainPrior.from_config(json.loads(f.attrs["grain_config"]))
        t_grid = f["t_grid"][:]
        b = int(f["block"][index])

    sl = spec.block_slice(b)
    seq = _seed_sequence(spec).spawn(spec.n_blocks)[b]
    out = draw_block(prior, grain_prior, t_grid, sl.stop - sl.start, spec.n_grains, seq)
    i = index - sl.start
    return {k: v[i] for k, v in out.items()}


# --------------------------------------------------------------------------
# Grain extension (PLAN.md proposed Stage 9a)
#
# More AHe grains per history is the one corpus change that is an *append*
# rather than a regeneration.  Grains are conditionally independent given the
# thermal history, and the history is already stored, so extra grains need no
# prior draw, no AFT rerun, and -- crucially -- leave every existing column,
# split and hyperparameter exactly where it was.  Extending the time span or the
# prior family cannot work this way: both change ``p(T | h)`` and therefore the
# paths themselves.
#
# The trap is the seed.  ``_seed_sequence`` is keyed on ``n_grains``, so simply
# re-running ``simulate`` with a larger one redraws *different histories* under
# the same filename.  The extension therefore gets its own lineage and the base
# ``spec`` is left untouched, describing what it always described: how the paths
# and the first ``spec.n_grains`` grains were drawn.
# --------------------------------------------------------------------------


def _extend_seed_sequence(spec: CorpusSpec, n_new: int) -> np.random.SeedSequence:
    """The extension's RNG lineage, disjoint from the base corpus's by construction.

    Two extra entropy words past :func:`_seed_sequence`, one of them a stream
    constant, so no choice of ``(seed, n_histories, n_grains)`` can make an
    extension collide with the draw that produced the histories it extends.
    """
    return np.random.SeedSequence(
        [spec.seed, spec.n_histories, spec.n_grains, n_new, EXTEND_STREAM])


def _spec_of(f: h5py.File) -> CorpusSpec:
    return CorpusSpec(**{k: tuple(v) if k == "split_fractions" else v
                         for k, v in json.loads(f.attrs["spec"]).items()})


def draw_extra_grains(grain_prior: GrainPrior, n: int, n_new: int,
                      seed_seq: np.random.SeedSequence) -> dict[str, np.ndarray]:
    """The latent half of an extension block: grain properties, no oracle.

    Split out for the same reason as :func:`draw_block` -- so the extension can
    be reproduced from the file alone without paying for RDAAM.
    """
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    grains = grain_prior.sample(n * n_new, rng)
    return {k: v.reshape(n, n_new) for k, v in grains.items()}


def _extend_block(args) -> tuple[int, dict[str, np.ndarray]]:
    """One block of extra grains on already-drawn histories.  Runs in a worker."""
    b, T, grain_cfg, n_new, t_grid, seq = args
    grain_prior = GrainPrior.from_config(grain_cfg)
    n = len(T)
    g = draw_extra_grains(grain_prior, n, n_new, seq)

    he = {k: np.empty((n, n_new), np.float32) for k in ("age", "corrected_age")}
    for i in range(n):
        for j in range(n_new):
            r = calculate_rdaam(t_grid, T[i], float(g["radius"][i, j]),
                                float(g["U"][i, j]), float(g["Th"][i, j]),
                                precision=PREC_BEST)
            he["age"][i, j] = r.age
            he["corrected_age"][i, j] = r.corrected_age

    out = {f"grain_{k}": v for k, v in g.items()}
    out["grain_eU"] = eU(g["U"], g["Th"])
    out["he_age"], out["he_corrected_age"] = he["age"], he["corrected_age"]
    # Same rule as the base corpus: flag, never drop.
    out["valid"] = (np.isfinite(he["age"]).all(axis=1) & (he["age"] > 0).all(axis=1)
                    & np.isfinite(he["corrected_age"]).all(axis=1))
    return b, out


def _create_extended(src: Path, dst: Path, n_new: int, compression: str | None) -> None:
    """Copy the corpus, widening the grain axis and leaving everything else alone.

    Written as a new file rather than an in-place resize because the base
    datasets were allocated without ``maxshape`` and because an interrupted
    in-place widening would leave the shipped corpus unreadable.  The source is
    opened read-only throughout and is never modified.
    """
    with h5py.File(src, "r") as fs, h5py.File(dst, "w") as fd:
        spec = _spec_of(fs)
        k0, n = spec.n_grains, spec.n_histories
        k = k0 + n_new
        chunk = min(spec.block_size, n)

        for name, value in fs.attrs.items():
            fd.attrs[name] = value
        for name in fs:
            if name != "he":
                fs.copy(name, fd)

        for name in ("age", "corrected_age", "radius", "U", "Th", "eU"):
            d = fd.create_dataset(f"he/{name}", shape=(n, k), dtype=np.float32,
                                  chunks=(chunk, k), compression=compression)
            d[:, :k0] = fs[f"he/{name}"][:]

        # ``valid`` becomes the conjunction over all k grains as blocks land, so
        # the untouched original is kept beside it: a row that the extension
        # invalidates must stay distinguishable from one the base corpus did.
        fd.create_dataset("valid_base", data=fs["valid"][:])
        fd.attrs["n_grains_total"] = k
        fd.attrs["extension"] = json.dumps({
            "n_new": n_new,
            "n_grains_total": k,
            "source": str(src),
            "source_git_sha": str(fs.attrs.get("git_sha", "unknown")),
            "source_created_utc": str(fs.attrs.get("created_utc", "unknown")),
            "seed_entropy": _extend_seed_sequence(spec, n_new).entropy,
            "stream": EXTEND_STREAM,
            **provenance(),
        })
        fd.create_dataset("grain_block_done", data=np.zeros(spec.n_blocks, dtype=bool))


def _store_extension(f: h5py.File, sl: slice, k0: int, out: dict[str, np.ndarray]) -> None:
    for name in ("age", "corrected_age", "radius", "U", "Th", "eU"):
        key = f"he_{name}" if name in ("age", "corrected_age") else f"grain_{name}"
        f[f"he/{name}"][sl, k0:] = out[key]
    # Idempotent: recomputed from the untouched base flag, so replaying a block
    # after an interruption cannot narrow ``valid`` twice.
    f["valid"][sl] = f["valid_base"][sl] & out["valid"]


def extend_grains(
    src: str | Path,
    dst: str | Path,
    n_new: int,
    *,
    grain_prior: GrainPrior | None = None,
    workers: int | None = None,
    compression: str | None = "lzf",
    progress: bool = True,
) -> Path:
    """Append ``n_new`` AHe grains per history to an existing corpus.

    Reads the stored thermal histories, draws fresh grain properties under the
    extension's own seed lineage, runs RDAAM on each, and writes a new corpus at
    ``dst`` with a grain axis of ``spec.n_grains + n_new``.  The source file is
    not modified.  Resumable on the same terms as :func:`simulate`: rerunning the
    identical call fills in whatever blocks are missing.

    The grain prior defaults to the one recorded in the source, which is what
    makes the appended grains exchangeable with the originals -- a training run
    that takes a random subset of the grain axis is then sampling one
    distribution, not a mixture of two.
    """
    import multiprocessing as mp

    src, dst = Path(src), Path(dst)
    if n_new < 1:
        raise ValueError(f"n_new must be >= 1, got {n_new}")

    with h5py.File(src, "r") as f:
        spec = _spec_of(f)
        cfg = json.loads(f.attrs["grain_config"])
        if int(np.sum(~f["block_done"][:])):
            raise ValueError(f"{src} is unfinished; finish it before extending it")
    grain_prior = grain_prior or GrainPrior.from_config(cfg)

    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        _create_extended(src, dst, n_new, compression)

    with h5py.File(dst, "r") as f:
        ext = json.loads(f.attrs["extension"])
        if ext["n_new"] != n_new:
            raise ValueError(f"{dst} was written for n_new={ext['n_new']}, not {n_new}")
        if _spec_of(f) != spec:
            raise ValueError(f"{dst} was extended from a different CorpusSpec")
        todo = np.flatnonzero(~f["grain_block_done"][:]).tolist()
        t_grid = f["t_grid"][:]
        T = f["T"][:]
    if not todo:
        return dst

    children = _extend_seed_sequence(spec, n_new).spawn(spec.n_blocks)
    jobs = [(b, T[spec.block_slice(b)], grain_prior.to_config(), n_new, t_grid, children[b])
            for b in todo]
    del T

    t0 = time.perf_counter()
    workers = workers or mp.cpu_count()
    with mp.Pool(workers) as pool, h5py.File(dst, "r+") as f:
        for done, (b, out) in enumerate(
                pool.imap_unordered(_extend_block, jobs, chunksize=1), start=1):
            _store_extension(f, spec.block_slice(b), spec.n_grains, out)
            f["grain_block_done"][b] = True
            f.flush()
            if progress and (done % 10 == 0 or done == len(jobs)):
                rate = done / (time.perf_counter() - t0)
                print(f"\r{done}/{len(jobs)} blocks · {rate * spec.block_size:.0f} "
                      f"histories/s · {(len(jobs) - done) / rate / 60:.1f} min left",
                      end="", flush=True)
    if progress:
        print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min")
    return dst


def reproduce_extra_grains(path: str | Path, index: int) -> dict[str, np.ndarray]:
    """Redraw the appended grain properties for ``index`` from the file alone.

    The counterpart of :func:`reproduce` for the extension's lineage.  Grain
    properties only: re-running RDAAM would defeat the purpose of it being cheap.
    """
    with h5py.File(path, "r") as f:
        spec = _spec_of(f)
        n_new = json.loads(f.attrs["extension"])["n_new"]
        grain_prior = GrainPrior.from_config(json.loads(f.attrs["grain_config"]))
        b = int(f["block"][index])

    sl = spec.block_slice(b)
    seq = _extend_seed_sequence(spec, n_new).spawn(spec.n_blocks)[b]
    out = draw_extra_grains(grain_prior, sl.stop - sl.start, n_new, seq)
    return {k: v[index - sl.start] for k, v in out.items()}
