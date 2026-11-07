# The precision limit of diffusion estimation at low CNR

The companion derivation (`docs/DERIVATION.md`) addresses *bias* — the
systematic error that makes ⟨D̂⟩ ≠ D. This note addresses the other half:
*variance* — how precise an unbiased estimate of D can possibly be. The two are
different in kind. Bias is a removable artefact of how the width estimator is
built. Variance, at low CNR, is a law: information the measurement physically
does not contain. No estimator, however clever, can beat it.

The bound is the Cramér–Rao lower bound (CRLB). It is a property of the
measurement model alone — it does not depend on the estimator — so it can be
computed independently of any derivation of an unbiased estimator, and it
measures how close to optimal an unbiased estimator is.

Computed and validated in `crlb.py` (this folder).

## 1. Fisher information of the measurement

At frame *i* the noise-free profile is `μ(x) = A_i exp(−x²/(2 v_i))`, with
`v_i = σ_0² + 2 D t_i` and `A_i` the peak amplitude. Each pixel carries
independent Gaussian noise of standard deviation σ_noise (known). For Gaussian
noise of known variance the Fisher information matrix is exactly

    F_{ab} = (1/σ_noise²) · Σ_{pixels, frames} (∂μ/∂θ_a)(∂μ/∂θ_b),

a sum over the model's mean-derivatives. The parameters are
`θ = (σ_0², D, A_1…A_N, c_1…c_N)`: two structural parameters and, per frame, a
nuisance amplitude `A_i` and centre `c_i`. For any unbiased estimator,

    Var(D̂) ≥ [F⁻¹]_{DD}.

## 2. Per-frame width information

The width enters only through `v_i`, and `∂μ/∂v = μ · x²/(2v²)`. Summing over a
dense pixel grid of spacing Δx (replacing the sum by an integral):

    width information, A and c known:
        J_i = (3√π / 16) · A_i² / (σ_noise² Δx) · v_i^(−3/2)

Two corrections:

- **The amplitude is unknown.** Profiling out the per-frame nuisance `A_i`
  inflates the width variance by exactly **3/2**. The effective per-frame width
  information is `J_i^eff = (2/3) J_i`, i.e.

        Var(σ̂_i²) ≥ 1 / J_i^eff = (8/√π) · σ_noise² Δx · v_i^(3/2) / A_i².

- **The centre is unknown — but free.** `∂μ/∂c` is odd in x and `∂μ/∂v` is
  even, so their information cross-term is zero. The unknown centre costs *no*
  width information. (An unknown baseline, by contrast, is even in x and would
  couple — a flat background does cost precision.)

In contrast-to-noise terms, with `CNR_i = A_i/σ_noise`:

    Var(σ̂_i²) ≥ (8/√π) · Δx · v_i^(3/2) / CNR_i².

The per-frame width variance floor scales as **1/CNR²** and grows as v^(3/2).
The continuum formula matches the exact discrete computation to 3 decimals
(`crlb.py`).

## 3. The bound on D

With `v_i = σ_0² + 2 D t_i`, the chain rule gives `∂μ/∂D = (∂μ/∂v)·2t_i` and
`∂μ/∂σ_0² = ∂μ/∂v`. The amplitudes `A_i` are frame-local, so profiling them out
simply replaces each frame's information by `J_i^eff`. What remains is a linear
two-parameter problem in (σ_0², D), and its inverse Fisher matrix gives

    Var(D̂) ≥ 1 / [ 4 · Σ_i J_i^eff · (t_i − t̄)² ],

where `t̄` is the `J_i^eff`-weighted mean of the frame times. This has the exact
form of a weighted straight-line-slope variance, with per-frame weights equal
to the inverse per-frame width CRLB.

**Consequence — the two-stage MSD pipeline is information-efficient.** The
expression above is *identical* to what you get by estimating each frame's
width independently at its own CRLB and then fitting the slope by weighted
least squares (verified numerically in `crlb.py`: the full-Fisher-matrix bound
and the per-frame-widths-plus-slope bound agree to 4 significant figures). The
"fit a width per frame, then fit the slope" architecture loses **no**
information. The low-CNR bias therefore comes entirely from the per-frame
width *estimator*, and the architecture adds none of it. Replace the biased width
fit with an unbiased, efficient one and weight the slope by the inverse CRLB,
and the diffusion estimate attains the fundamental limit.

## 4. The numbers

For the test regime — `L_D/FWHM_0 = 0.04`, so `D = 0.00444 σ_0²/τ`,
12 frames over 3 lifetimes — the CRLB gives the smallest standard deviation any
unbiased single-measurement estimate of D can have:

| CNR₀ | √CRLB / D | what it means |
|------|-----------|---------------|
| 3  | 24  | a single measurement cannot determine D at all |
| 5  | 15  | "" |
| 8  | 9.2 | "" |
| 15 | 4.9 | "" |
| 30 | 2.4 | "" |
| 80 | 0.9 | a single measurement is finally informative |

In the regime where nanocrystal-solid measurements sit, the *best possible*
unbiased estimate of D from one measurement has a standard deviation several
times larger than D itself. The limit is set by the information content of the
data and holds for every method. Validation: at high CNR, where the
least-squares fit is maximum-likelihood and attains the CRLB, the empirical
variance of the fitted width matches the bound to within 2–6 % (`crlb.py`).

## 5. Consequences

1. **Bias and variance are separate problems.** The bias is removable: the
   noise-bias correction (`docs/DERIVATION.md`) removes it. The variance is
   the CRLB and cannot be removed. Treating both as one problem ("low-CNR data
   are unreliable") hides that only the variance is fundamental.

2. **An unbiased estimator is what makes replication work.** Because the
   single-shot floor is several × D, a low-CNR diffusion coefficient can only be
   pinned down by averaging many measurements. Averaging only converges to the
   truth if each measurement is unbiased; averaging biased estimates converges
   to the wrong value. Together with the CRLB, an unbiased estimator is
   therefore what makes low-CNR diffusion measurement possible.
   The number of measurements needed scales as (√CRLB/D)² for unit relative
   precision — e.g. of order 200 at CNR₀ = 5.

3. **The CRLB is the benchmark for the corrected estimator.**
   `efficiency_check.py` compares the spread of the corrected D̂ with √CRLB on
   Gaussian, Voigt, and generalized-Gaussian (p = 1.5, p = 4) profiles. A fit of
   the corrected widths against time, with every frame weighted by
   1 / Var(σ̂_i²) and the intercept free, reaches 0.98 to 1.08 × √CRLB from
   CNR₀ 21 with no systematic shift of the mean; the library slope fit uses
   these weights. The remaining imprecision is the CRLB itself (see
   `README.md`, this folder).
