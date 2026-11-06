"""Sanity tests for width-debias.

Most are smoke tests that keep the moving parts wired together. The tests under
"Release guarantees" pin what the release notes promise: the subtraction itself,
the saved lookup's weights, worker-count independence, and the failed-fit
fallback. They import the installed package, not ``src/``, so CI tests what
ships.
"""
import numpy as np
import pytest

from width_debias import (AnomalousExperiment, Experiment, GaussianModel,
                            GeneralizedGaussianModel, MomentModel,
                            MultiExpExperiment, UnboundedGaussianModel,
                            VoigtModel, calibrate_d_free, evaluate,
                            frame_stats, model_from_spec, run_ensemble,
                            spec_tag)
from width_debias.fit import fit_D, fit_variance
from width_debias.simulate import noise_free_series


def test_noise_free_fit_recovers_variance():
    """A clean profile must fit back to its known variance."""
    exp = Experiment(D=0.0)
    profiles = noise_free_series(exp)
    ff = fit_variance(exp.x, profiles[0])
    assert ff.ok
    assert ff.variance == pytest.approx(exp.sigma0 ** 2, rel=1e-3)


def test_noise_free_fit_D_is_exact():
    """With no noise the linear MSD fit returns D exactly."""
    exp = Experiment(D=0.03)
    profiles = noise_free_series(exp)
    var = np.array([fit_variance(exp.x, row).variance for row in profiles])
    msd = var - var[0]
    d = fit_D(exp.times, msd, np.ones_like(msd))
    assert d.ok
    assert d.D == pytest.approx(exp.D, rel=1e-3)


def test_fit_D_covariance_se():
    """fit_D reports the weighted-LS covariance SE of D: it matches the closed
    form Var(slope)=Sw/(Sw*Swtt-Swt^2) and shrinks as 1/sqrt(precision)."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    msd = np.array([0.0, 0.04, 0.08, 0.12])   # slope 0.04 -> D = 0.02
    w = np.array([0.0, 1.0, 1.0, 1.0])        # t=0 carries no slope weight
    d = fit_D(t, msd, w)
    assert d.ok and np.isfinite(d.D_se) and d.D_se > 0
    good = w > 0
    sw = w[good].sum(); swt = (w[good] * t[good]).sum()
    swtt = (w[good] * t[good] ** 2).sum()
    assert d.D_se == pytest.approx(np.sqrt(sw / (sw * swtt - swt ** 2)) / 2,
                                   rel=1e-9)
    # 10x more precise data -> SE shrinks by sqrt(10).
    assert fit_D(t, msd, 10.0 * w).D_se == pytest.approx(d.D_se / np.sqrt(10),
                                                         rel=1e-6)


def test_var_bias_is_positive_and_grows_at_low_cnr():
    """The folded-normal prediction: sigma_hat^2 is biased high, more so at
    low CNR (DERIVATION.md s.2)."""
    exp = Experiment(D=0.02)
    lo = frame_stats(run_ensemble(exp, cnr0=4.0, n_runs=300, seed=0))
    hi = frame_stats(run_ensemble(exp, cnr0=60.0, n_runs=300, seed=0))
    assert np.nanmean(lo.var_bias) > 0.0
    assert np.nanmean(lo.var_bias) > np.nanmean(hi.var_bias)


def test_truncation_fraction_grows_at_low_cnr():
    """The wall bites harder at low CNR: p is larger when CNR_0 is small."""
    exp = Experiment(D=0.02)
    lo = frame_stats(run_ensemble(exp, cnr0=2.0, n_runs=300, seed=0))
    hi = frame_stats(run_ensemble(exp, cnr0=40.0, n_runs=300, seed=0))
    assert np.nanmean(lo.trunc_fraction) > np.nanmean(hi.trunc_fraction)


def test_low_cnr_inflates_D():
    """The known result: at low CNR the mean fitted D is biased high."""
    exp = Experiment(D=0.02)
    ens = run_ensemble(exp, cnr0=3.0, n_runs=400, seed=0)
    slopes = []
    for r in range(ens.n_runs):
        d = fit_D(exp.times, ens.msd[r], np.ones(exp.n_frames))
        if d.ok:
            slopes.append(d.D)
    assert np.mean(slopes) > exp.D  # positive (high) bias


@pytest.mark.parametrize("model", [
    GaussianModel(),
    VoigtModel(gamma=0.3),
    GeneralizedGaussianModel(shape=1.5),
    GeneralizedGaussianModel(shape=3.0),
    MomentModel(),
])
def test_models_recover_noise_free_variance(model):
    """Every estimator must recover the known variance from clean profiles."""
    exp = Experiment(D=0.02, n_frames=6, t_max=2.0)
    clean = model.noise_free_series(exp)
    for f in range(exp.n_frames):
        ff = model.fit_variance(exp.x, clean[f])
        assert ff.ok
        assert ff.variance == pytest.approx(exp.true_variance[f], rel=5e-3)


@pytest.mark.parametrize("model", [
    VoigtModel(gamma=0.3),
    GeneralizedGaussianModel(shape=1.5),
    MomentModel(),
])
def test_models_run_through_montecarlo(model):
    """Voigt, generalized-Gaussian and moment models run through the pipeline."""
    exp = Experiment(D=0.02, n_frames=8, t_max=2.0)
    ens = run_ensemble(exp, cnr0=20.0, n_runs=40, seed=0, model=model)
    assert ens.model_name == model.name
    assert ens.variance.shape == (40, exp.n_frames)
    assert np.isfinite(ens.cnr_frame).all()


def test_moment_estimator_can_go_negative():
    """The non-truncated moment estimator must admit negative variances at
    low CNR -- that is what makes it un-folded and (to leading order)
    unbiased."""
    exp = Experiment(D=0.02, n_frames=8, t_max=2.0)
    ens = run_ensemble(exp, cnr0=4.0, n_runs=300, seed=0, model=MomentModel())
    finite = ens.variance[np.isfinite(ens.variance)]
    assert np.any(finite < 0.0)


def test_unbounded_matches_bounded_at_high_cnr():
    """With the wall removed, the fit must be identical to the bounded one when
    the wall does not bind -- i.e. at healthy CNR. There the zero bound is moot,
    but the bias is not: both fits carry the same width-estimation bias, which
    the correction still removes."""
    exp = Experiment(D=0.0)
    clean = noise_free_series(exp)
    rng = np.random.default_rng(0)
    noisy = clean[0] + rng.normal(0.0, exp.amplitude0 / 30.0, exp.x.size)
    bnd = fit_variance(exp.x, noisy)
    unb = UnboundedGaussianModel().fit_variance(exp.x, noisy)
    assert unb.variance == pytest.approx(bnd.variance, rel=1e-3)
    assert not unb.at_wall   # the unbounded fit has no wall to pin against


def test_unbounded_recovers_noise_free_variance():
    """The unbounded fit must still recover the known variance from clean
    profiles -- removing the wall must not break the high-CNR estimator."""
    exp = Experiment(D=0.02, n_frames=6, t_max=2.0)
    model = UnboundedGaussianModel()
    clean = model.noise_free_series(exp)
    for f in range(exp.n_frames):
        ff = model.fit_variance(exp.x, clean[f])
        assert ff.ok
        assert ff.variance == pytest.approx(exp.true_variance[f], rel=5e-3)


def test_unbounded_admits_negative_variance_in_deep_noise():
    """The precision-fitted unbounded estimator must be *able* to return a
    negative variance on peakless data -- the property a variance-parameterised
    fit cannot have (sigma^2 = 0 is an uncrossable singularity). Negatives are
    rare even here (a nonlinear fit favours a positive noise bump), so this
    needs enough pure-noise trials to be a reliable smoke test, not a one-shot.
    """
    model = UnboundedGaussianModel()
    x = Experiment().x
    rng = np.random.default_rng(3)
    saw_negative = False
    for _ in range(400):
        ff = model.fit_variance(x, rng.normal(0.0, 1.0, x.size))
        if np.isfinite(ff.variance) and ff.variance < 0.0:
            saw_negative = True
            break
    assert saw_negative


def test_anomalous_reduces_to_fickian_at_alpha_one():
    """alpha = 1 must reproduce the Fickian Experiment exactly -- the
    consistency check the subdiffusive validation leans on."""
    fickian = Experiment(D=0.03)
    anomalous = AnomalousExperiment(D=0.03, alpha=1.0)
    assert np.allclose(anomalous.true_variance, fickian.true_variance)
    assert np.allclose(anomalous.true_msd, fickian.true_msd)


def test_anomalous_curvature_sign_matches_exponent():
    """MSD(t) = 2 D t^alpha is concave for alpha < 1 and convex for alpha > 1.
    That curvature is the disorder signature the correction must preserve."""
    t = AnomalousExperiment(alpha=0.6).times
    sub = AnomalousExperiment(D=0.02, alpha=0.6).true_msd
    sup = AnomalousExperiment(D=0.02, alpha=1.4).true_msd
    assert np.all(np.diff(sub[1:], 2) < 0)   # concave (drop the t=0 origin)
    assert np.all(np.diff(sup, 2) > 0)       # convex


def test_anomalous_noise_free_fit_recovers_imposed_variance():
    """The bed must actually produce profiles whose fitted width follows the
    imposed sigma^2(t) = sigma0^2 + 2 D t^alpha -- the premise of scoring the
    recovered alpha against truth."""
    exp = AnomalousExperiment(D=0.05, alpha=0.7, n_frames=8, t_max=3.0)
    clean = noise_free_series(exp)
    var = np.array([fit_variance(exp.x, row).variance for row in clean])
    assert np.allclose(var, exp.true_variance, rtol=5e-3)


def test_multiexp_single_component_reduces_to_single_exponential():
    """One component must reproduce the inherited single-exponential decay."""
    single = Experiment(tau=2.0)
    multi = MultiExpExperiment(tau=2.0, fractions=(1.0,), taus=(2.0,))
    assert np.allclose(multi.amplitude, single.amplitude)


def test_multiexp_normalised_to_initial_amplitude():
    """The summed decay is normalised so amplitude(0) = amplitude0, keeping the
    meaning of CNR_0 fixed across decay laws."""
    multi = MultiExpExperiment(amplitude0=1.0, fractions=(0.5, 0.5),
                               taus=(0.3, 3.0))
    assert multi.amplitude[0] == pytest.approx(1.0)
    assert np.all(np.diff(multi.amplitude) < 0)   # monotone decay


def test_multiexp_rejects_mismatched_components():
    """fractions and taus must pair up; a mismatch is a silent-bug risk."""
    with pytest.raises(ValueError):
        MultiExpExperiment(fractions=(0.5, 0.5), taus=(1.0,))


def test_generalized_gaussian_variance_factor():
    """The p=2 generalized Gaussian must reduce to the plain Gaussian."""
    g2 = GeneralizedGaussianModel(shape=2.0)
    assert g2.var_factor == pytest.approx(1.0, rel=1e-6)


def test_model_from_spec_sets_shape():
    """A shape in the spec must reach the model; the bare name is p = 2.

    The bare ``generalized_gaussian`` is a Gaussian profile, so a validation
    that means to test a non-Gaussian shape must carry the exponent through
    the spec string that worker processes rebuild the model from.
    """
    assert model_from_spec("generalized_gaussian").shape == 2.0
    heavy = model_from_spec("generalized_gaussian:1.5")
    assert isinstance(heavy, GeneralizedGaussianModel)
    assert heavy.shape == 1.5
    assert isinstance(model_from_spec("voigt"), VoigtModel)
    assert spec_tag("generalized_gaussian:4") == "generalized_gaussian_p4"
    assert spec_tag("gaussian") == "gaussian"
    with pytest.raises(ValueError):
        model_from_spec("voigt:1.5")
    with pytest.raises(ValueError):
        model_from_spec("lorentzian")


def test_evaluate_runs_and_returns_all_schemes():
    """The end-to-end harness produces a bias curve for every scheme."""
    exp = Experiment(D=0.02)
    grid = np.array([3.0, 10.0, 30.0])
    model = calibrate_d_free(grid, n_runs=150, seed=1)
    results = evaluate(exp, grid, model, n_runs=150, seed=5000)
    assert set(results) == {"ols", "inverse_variance", "subtract", "mas"}
    for r in results.values():
        assert r.mean_bias.shape == grid.shape
        assert np.all(np.isfinite(r.mean_bias))


def test_subtract_beats_inverse_variance_at_low_cnr():
    """The derived scheme should leave less D-bias than inverse-variance in
    the low-CNR regime where the noise bias is large, with the recommended
    D-free calibration (built at D = 0, applied at D = 0.02)."""
    exp = Experiment(D=0.02)
    grid = np.array([6.0, 10.0, 18.0, 32.0])
    model = calibrate_d_free(grid, n_runs=300, seed=1)
    results = evaluate(exp, grid, model, n_runs=300, seed=5000)
    iv = np.nanmean(np.abs(results["inverse_variance"].mean_bias - 1.0))
    fn = np.nanmean(np.abs(results["subtract"].mean_bias - 1.0))
    assert fn < iv


def test_interp_se_matches_hand_calculation():
    """The 1-D lookup SE uses the same interpolation weights as the lookup."""
    from width_debias.bias_lookup import _interp_se
    xp = np.array([1.0, 2.0, 4.0])
    sp = np.array([0.3, 0.4, 0.1])
    # x = 1.5 sits halfway between points 0 and 1.
    assert _interp_se(1.5, xp, sp) == pytest.approx(np.sqrt(0.25 * 0.09 + 0.25 * 0.16))
    # Outside the grid the edge point's SE applies (np.interp clamps).
    assert _interp_se(10.0, xp, sp) == pytest.approx(0.1)


def test_calibration_se_fields_populated_and_absent():
    """frame_stats -> from_stats carries var_bias SE; a bare model returns NaN."""
    from width_debias import BiasLookup, calibrate_d_free
    trunc = calibrate_d_free(np.array([5.0, 20.0]), n_runs=60, seed=3)
    assert trunc.varbias_se_grid is not None
    assert trunc.varbias_se_grid.shape == trunc.varbias_grid.shape
    assert np.all(trunc.varbias_se_grid[np.isfinite(trunc.varbias_se_grid)] >= 0)
    bare = BiasLookup(cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
                           relvar_grid=trunc.relvar_grid,
                           varbias_grid=trunc.varbias_grid)
    assert np.isnan(bare.var_bias_se(3.0))


def test_frame_variance_weights_do_not_spike_at_calibration_cnr0():
    """The 1-D weights use per-frame Var(sigma_hat^2), not the MSD-point lookup.

    The MSD-point lookup has a 0 at every calibration CNR_0 (MSD(0) = 0 by
    construction), so a frame next to one drew a huge weight. The per-frame
    grid has no zeros, and the weights it gives grow smoothly with CNR.
    """
    from dataclasses import replace
    from width_debias import calibrate_d_free, msd_and_weights
    trunc = calibrate_d_free(np.array([5.0, 20.0]), n_runs=60, seed=3)
    assert trunc.framevar_grid is not None
    assert trunc.framevar_grid.shape == trunc.cnr_grid.shape
    assert np.all(trunc.framevar_grid > 0)
    assert np.any(trunc.relvar_grid == 0)
    # Frames of a CNR_0 = 20 measurement; one sits next to the CNR_0 = 5 point.
    cnr = 20.0 * np.exp(-Experiment().times)
    k = np.argmin(np.abs(cnr - 5.0))
    cnr[k] = 5.0 * 1.001
    variance_row = np.ones_like(cnr)
    _, w_new = msd_and_weights("subtract", variance_row, cnr, trunc)
    _, w_old = msd_and_weights("subtract", variance_row, cnr,
                               replace(trunc, framevar_grid=None))
    neighbours = w_new[[k - 1, k + 1]]
    assert neighbours.min() < w_new[k] < neighbours.max()
    assert w_old[k] > 3 * w_old[[k - 1, k + 1]].max()


def test_model_without_frame_variance_keeps_msd_point_weights():
    """A model saved before framevar_grid existed weights by rel_variance."""
    from width_debias import BiasLookup, calibrate_d_free, msd_and_weights
    trunc = calibrate_d_free(np.array([5.0, 20.0]), n_runs=60, seed=3)
    bare = BiasLookup(cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
                           relvar_grid=trunc.relvar_grid,
                           varbias_grid=trunc.varbias_grid)
    assert bare.frame_variance(3.0) is None
    cnr = 12.0 * np.exp(-Experiment().times)
    _, w = msd_and_weights("inverse_variance", np.ones_like(cnr), cnr, bare)
    expected = 1.0 / bare.rel_variance(cnr)
    assert w[0] == 0.0
    assert np.allclose(w[1:], expected[1:])


def test_weighted_msd_fit_equals_width_fit():
    """With frame 0 weighted, the MSD slope is the slope of the corrected widths.

    A constant shift (sigma_hat^2(0)) does not change a free-intercept slope,
    so fit_D on the MSD returns the same D and covariance SE as a weighted fit
    of the independent corrected widths against time.
    """
    from width_debias import calibrate_d_free, corrected_variance, msd_and_weights
    trunc = calibrate_d_free(np.array([5.0, 20.0]), n_runs=60, seed=3)
    exp = Experiment(D=0.02)
    ens = run_ensemble(exp, 12.0, n_runs=5, seed=11)
    for row in ens.variance:
        msd, w = msd_and_weights("subtract", row, ens.cnr_frame, trunc)
        assert w[0] > 0
        assert np.allclose(w, 1.0 / trunc.frame_variance(ens.cnr_frame))
        widths = corrected_variance("subtract", row, ens.cnr_frame, trunc)
        a, b = fit_D(exp.times, msd, w), fit_D(exp.times, widths, w)
        assert a.D == pytest.approx(b.D, rel=1e-10)
        assert a.D_se == pytest.approx(b.D_se, rel=1e-12)


def test_calibration_D_se_predicts_draw_to_draw_scatter():
    """The propagated calibration SE matches the scatter of corrected D across
    independent calibration draws applied to one fixed evaluation set.

    The calibration error is common to every run evaluated with that
    calibration, so averaging runs does not remove it; the spread of the
    run-averaged D across draws isolates it.
    """
    from width_debias import calibrate_d_free, calibration_D_se, msd_and_weights
    exp = Experiment(D=0.02)
    ens = run_ensemble(exp, 13.0, n_runs=60, seed=777)
    means, predicted = [], []
    for k in range(8):
        # sweep_stats seeds CNR_0 cell i with seed + i, so draws spaced by 1000
        # share no ensemble; adjacent seeds (100 + k) would share two of three
        # cells and make the draws correlated, shrinking the observed scatter.
        trunc = calibrate_d_free(np.array([6.0, 13.0, 30.0]), n_runs=80,
                                 seed=1000 * (k + 1))
        ds, ses = [], []
        for r in range(ens.n_runs):
            msd, w = msd_and_weights("subtract", ens.variance[r],
                                     ens.cnr_frame, trunc)
            d = fit_D(exp.times, msd, w)
            if d.ok:
                ds.append(d.D)
                ses.append(calibration_D_se(exp.times, msd, w, ens.cnr_frame, trunc))
        means.append(np.mean(ds))
        predicted.append(np.nanmean(ses))
    observed = np.std(means, ddof=1)
    ratio = observed / np.mean(predicted)
    assert 0.5 < ratio < 2.0, (observed, np.mean(predicted))


# --------------------------------------------------------------------------
# Fisher-anchored calibration
# --------------------------------------------------------------------------
def _synthetic_width_scaled_model(bias_coef=0.05):
    """A model whose calibrated bias is exactly bias_coef / CNR^2, with a sqrt(v) width table."""
    from width_debias import BiasLookup
    grid = np.geomspace(0.5, 1000.0, 400)
    variance_grid = np.geomspace(0.25, 16.0, 41)
    return BiasLookup(cnr_grid=grid, p_grid=np.zeros_like(grid),
                           relvar_grid=np.ones_like(grid),
                           varbias_grid=bias_coef / grid ** 2,
                           width_variance_grid=variance_grid,
                           width_scale_grid=np.sqrt(variance_grid),
                           calibration_variance=1.0)


def test_fisher_leading_bias_matches_closed_form():
    """1-D Gaussian, peak CNR, white noise: Var(sigma_hat) = (2/sqrt(pi)) sigma dx / CNR^2."""
    from width_debias import leading_order_terms
    x = Experiment().x
    terms = leading_order_terms(GaussianModel(), x, 1.0)
    assert terms.bias_coef == pytest.approx(2.0 / np.sqrt(np.pi) * (x[1] - x[0]), rel=1e-3)
    assert terms.framevar_coef == pytest.approx(4.0 * terms.bias_coef, rel=1e-12)


def test_fisher_numerical_jacobian_matches_analytic_gaussian():
    from width_debias.fisher import var_of_fitted_variance
    x = Experiment().x
    g = np.exp(-0.5 * x ** 2)
    jac = np.stack([x ** 2 / 2.0 * g, g, x * g])      # d/dv, d/dA, d/dc at v = 1
    analytic = np.linalg.inv(jac @ jac.T)[0, 0]
    numerical = var_of_fitted_variance(GaussianModel(), x, 1.0, 1.0, estimator="ml")
    assert numerical == pytest.approx(analytic, rel=1e-5)


def test_least_squares_sandwich_equals_crlb_only_for_white_noise():
    from width_debias.fisher import var_of_fitted_variance
    x = Experiment().x
    model = GaussianModel()
    ls = var_of_fitted_variance(model, x, 1.0, 2.0, estimator="ls")
    ml = var_of_fitted_variance(model, x, 1.0, 2.0, estimator="ml")
    assert ls == pytest.approx(ml, rel=1e-10)
    shot = 1.0 + 5.0 * np.exp(-0.5 * x ** 2)            # noise concentrated on the peak
    assert var_of_fitted_variance(model, x, 1.0, shot, estimator="ls") > \
        1.001 * var_of_fitted_variance(model, x, 1.0, shot, estimator="ml")


def test_width_scale_grows_as_sigma_and_faster_once_the_window_truncates():
    from width_debias import leading_order_terms
    variance_grid = np.array([0.25, 0.5, 1.0, 1.5, 2.25, 9.0])
    terms = leading_order_terms(GaussianModel(), Experiment().x, 1.0, variance_grid)
    inside = variance_grid <= 2.25
    assert np.allclose(terms.bias_scale[inside], np.sqrt(variance_grid[inside]), rtol=1e-3)
    assert terms.bias_scale[-1] > 1.01 * np.sqrt(9.0)


def test_pspline_recovers_curve_and_is_flat_above_knee():
    from width_debias.smoothing import pspline_fit
    rng = np.random.default_rng(0)
    knee = 1.8
    u = np.linspace(-1.0, 3.5, 80)
    truth = np.where(u < knee, 0.3 * (knee - u) ** 2, 0.0)
    se = np.full(u.size, 0.02)
    fit = pspline_fit(u, truth + rng.normal(0.0, 0.02, u.size), se,
                      knee=knee, intercept=False)
    assert np.max(np.abs(fit.evaluate(u) - truth)) < 0.04
    assert np.all(fit.evaluate(np.array([knee, 2.5, 3.5])) == 0.0)
    assert np.linalg.eigvalsh(fit.coef_cov).min() > -1e-12


def test_width_scaled_lookup_inverts_the_bias_at_the_true_width():
    model = _synthetic_width_scaled_model(bias_coef=0.05)
    cnr, v_true = 3.0, 4.0
    bias = np.sqrt(v_true) * 0.05 / cnr ** 2
    assert model.var_bias(cnr, v_true + bias) == pytest.approx(bias, rel=1e-3)
    assert model.var_bias(cnr) == pytest.approx(0.05 / cnr ** 2, rel=1e-3)


def test_correlated_calibration_se_reduces_to_independent_for_diagonal_factor():
    from dataclasses import replace
    from width_debias import BiasLookup, calibration_D_se
    grid = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    se = np.array([0.03, 0.02, 0.01, 0.005, 0.002])
    independent = BiasLookup(cnr_grid=grid, p_grid=np.zeros(5),
                                  relvar_grid=np.ones(5), varbias_grid=0.05 / grid ** 2,
                                  varbias_se_grid=se)
    correlated = replace(independent, varbias_cov_factor=np.diag(se))
    assert np.allclose(correlated.var_bias_se(grid), se)
    times = np.linspace(0.0, 2.0, 5)
    cnr = grid[::-1]
    msd, w = np.zeros(5), np.ones(5)
    assert calibration_D_se(times, msd, w, cnr, correlated) == pytest.approx(
        calibration_D_se(times, msd, w, cnr, independent), rel=1e-12)


def test_fisher_calibration_cuts_calibration_se_at_equal_budget():
    """The same number of Monte Carlo fits gives a smaller calibration SE of D."""
    from width_debias import calibrate_d_free, calibrate_fisher, calibration_D_se, msd_and_weights
    raw = calibrate_d_free(np.array([5.0, 20.0]), n_runs=60, seed=3)     # 2 x 60 x 12 fits
    smoothed = calibrate_fisher(np.geomspace(0.5, 30.0, 24), n_runs=60, seed=4)  # 24 x 60
    exp = Experiment(D=0.02)
    ens = run_ensemble(exp, 13.0, n_runs=20, seed=777)

    def mean_calibration_se(model):
        ses = []
        for r in range(ens.n_runs):
            msd, w = msd_and_weights("subtract", ens.variance[r], ens.cnr_frame, model)
            ses.append(calibration_D_se(exp.times, msd, w, ens.cnr_frame, model, ens.variance[r]))
        return np.nanmean(ses)

    assert mean_calibration_se(smoothed) < 0.8 * mean_calibration_se(raw)
    assert smoothed.var_bias(20.0) == pytest.approx(smoothed.var_bias(20.0, 1.3))  # no width table


def test_new_fields_leave_old_lookups_unchanged():
    from dataclasses import replace
    from width_debias import calibrate_d_free
    trunc = calibrate_d_free(np.array([5.0, 20.0]), n_runs=60, seed=3)
    cnr = np.array([0.7, 2.0, 9.0])
    plain = np.interp(cnr, trunc.cnr_grid, trunc.varbias_grid)
    assert np.array_equal(trunc.var_bias(cnr, np.array([1.2, 0.8, 3.0])), plain)
    scaled = _synthetic_width_scaled_model()
    stripped = replace(scaled, width_scale_grid=None, width_variance_grid=None,
                       calibration_variance=None)
    assert np.array_equal(stripped.var_bias(cnr, np.array([4.0, 4.0, 4.0])),
                          stripped.var_bias(cnr))
    assert np.array_equal(trunc.frame_variance(cnr),
                          np.interp(cnr, trunc.cnr_grid, trunc.framevar_grid))


def test_bias_lookup_save_load_round_trip(tmp_path):
    from width_debias import BiasLookup, calibrate_d_free
    scaled = _synthetic_width_scaled_model()
    scaled.save(tmp_path / "scaled.npz")
    with pytest.warns(UserWarning, match="framevar_grid"):
        loaded = BiasLookup.load(tmp_path / "scaled.npz")
    assert loaded.calibration_variance == 1.0
    assert loaded.var_bias(3.0, 4.0) == pytest.approx(scaled.var_bias(3.0, 4.0), rel=1e-15)
    trunc = calibrate_d_free(np.array([5.0, 20.0]), n_runs=60, seed=3)
    trunc.save(tmp_path / "plain.npz")
    with np.load(tmp_path / "plain.npz") as archive:      # the experiments' loader
        legacy = BiasLookup(**{k: archive[k] for k in archive.files})
    assert not legacy.is_2d and not legacy.is_width_scaled
    assert np.array_equal(legacy.var_bias(np.array([1.0, 4.0])),
                          trunc.var_bias(np.array([1.0, 4.0])))


def test_load_accepts_the_path_given_to_save_and_releases_the_file(tmp_path, dfree_lookup):
    """save("cal") writes cal.npz; load("cal") must find it, and must close the
    archive so the file can be replaced or removed (Windows locks open files)."""
    from width_debias import BiasLookup
    from width_debias.bias_lookup import LOOKUP_FORMAT_VERSION
    stem = tmp_path / "cal"
    dfree_lookup.save(stem)
    loaded = BiasLookup.load(stem)
    assert loaded.format_version == LOOKUP_FORMAT_VERSION
    assert np.array_equal(loaded.varbias_grid, dfree_lookup.varbias_grid)
    (tmp_path / "cal.npz").unlink()


def test_load_refuses_a_newer_archive_format(tmp_path, dfree_lookup):
    from dataclasses import replace
    from width_debias import BiasLookup
    from width_debias.bias_lookup import LOOKUP_FORMAT_VERSION
    path = tmp_path / "future.npz"
    arrays = {k: v for k, v in vars(replace(dfree_lookup)).items() if v is not None}
    arrays["format_version"] = LOOKUP_FORMAT_VERSION + 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="archive format"):
        BiasLookup.load(path)


# --------------------------------------------------------------------------
# Release guarantees
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dfree_lookup():
    """A small D-free calibration shared by the guarantee tests."""
    return calibrate_d_free(np.array([5.0, 20.0]), n_runs=60, seed=3)


def test_subtract_removes_exactly_the_looked_up_bias(dfree_lookup):
    """``subtract`` is sigma_hat^2 - var_bias(CNR), with no rescaling, and the
    MSD is the corrected widths less the corrected t = 0 width."""
    from width_debias import corrected_variance, msd_and_weights
    cnr = 15.0 * np.exp(-Experiment().times)
    variance_row = 1.0 + 0.03 * Experiment().times
    expected = variance_row - np.interp(cnr, dfree_lookup.cnr_grid,
                                        dfree_lookup.varbias_grid)
    corrected = corrected_variance("subtract", variance_row, cnr, dfree_lookup)
    assert np.array_equal(corrected, expected)
    msd, _ = msd_and_weights("subtract", variance_row, cnr, dfree_lookup)
    assert np.array_equal(msd, expected - expected[0])
    for scheme in ("ols", "inverse_variance"):
        assert np.array_equal(
            corrected_variance(scheme, variance_row, cnr, dfree_lookup), variance_row)


def test_calibrated_bias_is_the_ensemble_mean_less_the_truth():
    """The lookup stores E[sigma_hat^2] - sigma^2_true per calibration frame,
    over every finite fit, with no rescaling.

    A statistical check of the subtraction cannot resolve a wrong scale at a
    unit-test budget (at CNR_0 5 the combined SE is about the size of the bias),
    so the calibration side is pinned exactly here and the application side in
    test_subtract_removes_exactly_the_looked_up_bias.
    """
    from width_debias import BiasLookup
    exp = Experiment(D=0.0)
    ens = run_ensemble(exp, 5.0, n_runs=30, seed=21)
    expected = np.nanmean(ens.variance, axis=0) - exp.true_variance
    stats = frame_stats(ens)
    assert np.allclose(stats.var_bias, expected, rtol=0, atol=1e-15)
    lookup = BiasLookup.from_stats([stats])
    assert np.allclose(lookup.var_bias(ens.cnr_frame), expected, rtol=0, atol=1e-15)


def test_saved_lookup_keeps_frame_variance_and_weights(tmp_path, dfree_lookup):
    """A save/load round trip keeps framevar_grid, so the weights do not fall
    back to the MSD-point variance."""
    from width_debias import BiasLookup, msd_and_weights
    dfree_lookup.save(tmp_path / "cal.npz")
    loaded = BiasLookup.load(tmp_path / "cal.npz")
    assert loaded.framevar_grid is not None
    assert np.array_equal(loaded.framevar_grid, dfree_lookup.framevar_grid)
    cnr = 12.0 * np.exp(-Experiment().times)
    row = np.linspace(1.0, 1.1, cnr.size)
    for scheme in ("subtract", "inverse_variance", "mas"):
        msd_a, w_a = msd_and_weights(scheme, row, cnr, dfree_lookup)
        msd_b, w_b = msd_and_weights(scheme, row, cnr, loaded)
        assert np.array_equal(msd_a, msd_b)
        assert np.array_equal(w_a, w_b)
        assert w_b[0] > 0


def test_results_do_not_depend_on_worker_count():
    """Each realisation is seeded on its own, so serial and parallel agree."""
    exp = Experiment(D=0.02, n_frames=6)
    serial = run_ensemble(exp, 10.0, n_runs=6, seed=5, workers=1)
    parallel = run_ensemble(exp, 10.0, n_runs=6, seed=5, workers=2)
    assert np.array_equal(serial.variance, parallel.variance)
    assert np.array_equal(serial.cnr_frame, parallel.cnr_frame)


def test_failed_fit_falls_back_and_is_kept(monkeypatch):
    """A fit that fails returns the broad initial guess, flagged not ok; it is
    never dropped (dropping failures would delete the tail that carries the bias)."""
    import width_debias.fit as fit_module

    def fail(*args, **kwargs):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(fit_module, "curve_fit", fail)
    x = Experiment().x
    ff = fit_module.fit_variance(x, np.exp(-0.5 * x ** 2))
    span = x[-1] - x[0]
    assert not ff.ok
    assert ff.variance == pytest.approx((span / 4.0) ** 2)


def test_fit_D_drops_nan_frames():
    """NaN marks an unusable frame; fit_D leaves it out instead of failing."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    msd = np.array([0.0, 0.04, np.nan, 0.12])
    d = fit_D(t, msd, np.ones(4))
    assert d.ok
    assert d.D == pytest.approx(0.02, rel=1e-9)


def test_one_d_lookup_ignores_the_width_key(dfree_lookup):
    """A 1-D lookup with no width table gives the same correction and weights
    with or without a width key."""
    from width_debias import msd_and_weights
    cnr = 12.0 * np.exp(-Experiment().times)
    row = np.linspace(1.0, 1.1, cnr.size)
    key = np.full(cnr.size, 2.5)
    msd_a, w_a = msd_and_weights("subtract", row, cnr, dfree_lookup)
    msd_b, w_b = msd_and_weights("subtract", row, cnr, dfree_lookup, key_variance=key)
    assert np.array_equal(msd_a, msd_b)
    assert np.array_equal(w_a, w_b)


def test_per_frame_cnr_estimates_one_cnr_per_frame():
    """fft-cnr returns a finite CNR per profile, close to the true peak CNR."""
    pytest.importorskip("fft_cnr")
    from width_debias.cnr import per_frame_cnr
    x = Experiment().x
    rng = np.random.default_rng(0)
    window = np.abs(x) <= 4.0
    profiles = np.array([np.exp(-0.5 * x[window] ** 2) + rng.normal(0, 1 / 20.0, window.sum())
                         for _ in range(4)])
    cnr = per_frame_cnr(profiles)
    assert cnr.shape == (4,)
    assert np.all(np.isfinite(cnr))
    assert 0.7 * 20.0 < np.median(cnr) < 1.4 * 20.0


# --------------------------------------------------------------------------
# Experimental calibrations
# --------------------------------------------------------------------------
def test_joint_calibration_fits_with_the_model_passed():
    """Each cell fits with the instance passed, so the shape exponent is kept."""
    from width_debias import calibrate_cnr_sigma2
    kwargs = dict(cnr_grid=np.array([3.0]), variance_grid=np.array([1.0]),
                  n_runs=40, seed=2, n_workers=1)
    flat = calibrate_cnr_sigma2(GeneralizedGaussianModel(shape=4.0), **kwargs)
    heavy = calibrate_cnr_sigma2(GeneralizedGaussianModel(shape=1.5), **kwargs)
    assert flat.var_bias[0, 0] != heavy.var_bias[0, 0]


def test_joint_calibration_keeps_every_fit_by_default():
    """The default bias is the mean over every finite fit, as calibrate_cells
    computes it; only an explicit threshold drops the high fits."""
    from width_debias import calibrate_cnr_sigma2
    from width_debias.calibration import _cell_fits
    x = Experiment().x
    model = GaussianModel()
    fits = _cell_fits(model, x, 0.8, 1.0, 1.0, 60, np.random.SeedSequence(9).spawn(1)[0],
                      "gaussian", 0.0, 0.0, 1.0)
    grid = calibrate_cnr_sigma2(model, pixel_x=x, cnr_grid=np.array([0.8]),
                                variance_grid=np.array([1.0]), n_runs=60, seed=9,
                                n_workers=1)
    assert grid.var_bias[0, 0] == pytest.approx(np.mean(fits[np.isfinite(fits)]) - 1.0,
                                                rel=1e-12)
    capped = calibrate_cnr_sigma2(model, pixel_x=x, cnr_grid=np.array([0.8]),
                                  variance_grid=np.array([1.0]), n_runs=60, seed=9,
                                  n_workers=1, filter_threshold=5.0)
    assert capped.var_bias[0, 0] <= grid.var_bias[0, 0]


@pytest.mark.parametrize("noise_model", ["shot", "white", None])
def test_joint_calibration_refuses_unknown_noise_laws(noise_model):
    from width_debias import calibrate_cnr_sigma2
    from width_debias.noise import ShotNoise
    with pytest.raises((ValueError, TypeError)):
        calibrate_cnr_sigma2(cnr_grid=np.array([3.0]), variance_grid=np.array([1.0]),
                             n_runs=2, n_workers=1, noise_model=noise_model)
    with pytest.raises(TypeError):
        calibrate_cnr_sigma2(cnr_grid=np.array([3.0]), variance_grid=np.array([1.0]),
                             n_runs=2, n_workers=1, noise_model=ShotNoise())


def test_calibration_profile_and_fisher_terms_do_not_assume_a_centred_grid():
    """A pixel grid from 0 to 12 holds the profile at its midpoint: the clean
    profile fits back to its true variance, and the Fisher terms match the
    centred grid's (they are translation invariant)."""
    from width_debias import leading_order_terms
    from width_debias.calibration import _single_frame_profile
    centred = Experiment().x
    shifted = centred - centred[0]
    clean = _single_frame_profile(GaussianModel(), shifted, 1.0, 1.0)
    assert fit_variance(shifted, clean).variance == pytest.approx(1.0, rel=1e-3)
    a = leading_order_terms(GaussianModel(), centred, 1.0)
    b = leading_order_terms(GaussianModel(), shifted, 1.0)
    assert b.bias_coef == pytest.approx(a.bias_coef, rel=1e-6)
