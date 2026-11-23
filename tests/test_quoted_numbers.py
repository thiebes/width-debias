"""Numbers quoted in the documentation match the committed tables they come from.

The validation numbers are restated by hand in the README, docs/DERIVATION.md,
docs/REPO_MAP.md, set READMEs, captions and library docstrings, and a rerun
that changes a table leaves the copies behind. Each claim below recomputes a
quoted value from the committed ``.txt`` table that produced it and checks the
copies it lists. Only listed copies are checked: a copy that no claim lists is
not found. When a rerun changes a table, a claim fails until each listed copy,
and its snippet here, are updated; when a copy is reworded, its snippet fails
until it is updated. ``RETIRED`` lists superseded values that must not
reappear in a tracked document other than the changelog.

A copy matches when its snippet occurs in the document, not inside a longer
number ("4 of 140" does not match "14 of 140"), and the claim's numbers appear
in the snippet in order and side by side. Before matching, both texts collapse
whitespace runs (so line breaks and docstring indentation do not matter), read
"±" as "+/-" and the Unicode minus as a hyphen. Numbers are compared as
printed, with a leading plus sign dropped, so "+0.004 ± 0.008" and
"0.004 +/- 0.008" both carry ("0.004", "0.008").
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from functools import cache
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
DFREE = "experiments/dfree-calibration"
FLOOR = "experiments/correction-floor"
ESTIMATOR = "experiments/estimator-comparison"
COVERAGE = "experiments/coverage-interval"
HALIDE = "experiments/halide-mixing-highn"
DFREE_SHAPES = ("gaussian", "voigt", "genGauss_p1.5", "genGauss_p4")
POC_SHAPES = ("gaussian", "voigt", "generalized_gaussian_p1.5", "generalized_gaussian_p4")
ANALYTICAL_SHAPES = ("gaussian", "voigt", "generalized_gaussian_p1.5",
                     "generalized_gaussian_p4")
EFFICIENCY_SPECS = ("gaussian", "voigt", "generalized_gaussian:1.5", "generalized_gaussian:4")
HISTORY_DOCUMENTS = frozenset({"CHANGELOG.md"})

# A number not glued to a preceding word or digit: in "0.6-1.0" the hyphen is a
# range dash, not a sign, and "sigma_0" carries no number.
NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?")
NOT_AFTER_NUMBER = r"(?<![\w.])"
NOT_BEFORE_NUMBER = r"(?!\.?\d)"


@dataclass(frozen=True)
class Claim:
    """One quoted value and its copies.

    ``compute`` returns the value as the copies print it: strings, in the order
    the copies give them, with no plus sign. Table values are read as Decimal
    so they keep the table's digits until a reader rounds them.
    """

    name: str
    compute: Callable[[], tuple[str, ...]]
    path_to_snippets: dict[str, tuple[str, ...]]


# --------------------------------------------------------------------------
# Reading documents and tables
# --------------------------------------------------------------------------
@cache
def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n")


def _flat(text: str) -> str:
    """Whitespace runs collapsed, "±" read as "+/-", the Unicode minus as a hyphen."""
    return " ".join(text.replace("±", "+/-").replace("−", "-").split())


def _contains(document: str, snippet: str) -> bool:
    """Whether the snippet occurs in the document, not inside a longer number."""
    flat = _flat(snippet)
    pattern = re.escape(flat)
    if re.match(r"[-+.\d]", flat):
        pattern = NOT_AFTER_NUMBER + pattern
    if flat[-1].isdigit():
        pattern += NOT_BEFORE_NUMBER
    return re.search(pattern, _flat(document)) is not None


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(m.replace("−", "-").lstrip("+") for m in NUMBER.findall(text))


def _rounded(value: Decimal, places: int) -> str:
    return str(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _bound(value: Decimal, places: int) -> str:
    """The smallest number at ``places`` decimals not below ``value``, as a
    "within X" statement quotes it."""
    return str(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_CEILING))


def _section(text: str, start: str, next_start: str, source: str) -> str:
    """From the line beginning ``start`` to the next line beginning ``next_start``."""
    begin = text.find(start)
    if begin < 0:
        raise AssertionError(f"{source}: no section starting {start!r}")
    end = text.find("\n" + next_start, begin + len(start))
    return text[begin:] if end < 0 else text[begin:end]


def _search(pattern: str, text: str, source: str) -> re.Match:
    match = re.search(pattern, text, flags=re.M)
    if match is None:
        raise AssertionError(f"{source}: pattern {pattern!r} not found")
    return match


def _first(pattern: str, text: str, source: str) -> Decimal:
    return Decimal(_search(pattern, text, source).group(1))


# --------------------------------------------------------------------------
# Table readers, one per kind of claim
# --------------------------------------------------------------------------
def _dfree_arm(shape: str, window: str, arm: str) -> tuple[str, str]:
    """One arm's section of a D-free validation table, and the table's path."""
    stem = "validate_d_zero" if window == "standard" else "validate_d_zero_short"
    path = f"{DFREE}/{stem}_{shape}.txt"
    return _section(_read(path), f"--- {arm}\n", "--- ", path), path


def _cells_beyond(window: str, arm: str) -> tuple[str, ...]:
    """Cells beyond two combined SE, summed over the four shapes, as N and M of "N of M"."""
    counts = [_search(r"cells beyond 2 combined SE: (\d+) of (\d+)",
                      *_dfree_arm(shape, window, arm)).groups() for shape in DFREE_SHAPES]
    return str(sum(int(n) for n, _ in counts)), str(sum(int(m) for _, m in counts))


def _dfree_pooled_bound() -> tuple[str, ...]:
    """Largest |pooled z| of the corrected arm (CNR >= 1) over the four shapes."""
    z = max(abs(_first(r"all cells: .*; ([-+]?[\d.]+) combined SE",
                       *_dfree_arm(shape, "standard", "corrected, CNR >= 1")))
            for shape in DFREE_SHAPES)
    return (_bound(z, 1),)


def _dense_pooled_bound() -> tuple[str, ...]:
    """Largest |pooled z| of ``subtract`` over all test points of the four shapes.

    Each shape's pooled line is read inside that shape's ``subtract`` block, so
    a renamed or missing line fails here instead of reading the MAS block.
    """
    path = f"{DFREE}/dense_calibration_and_test_four_shapes.txt"
    shape_blocks = re.split(r"^=== ", _read(path), flags=re.M)[1:]
    assert len(shape_blocks) == 4, f"{path}: expected 4 shape blocks, found {len(shape_blocks)}"
    z = max(abs(_first(r"all points .*; ([-+]?[\d.]+) SE",
                       _section(block, "  subtract: pooled", "  mas: pooled", path), path))
            for block in shape_blocks)
    return (_bound(z, 1),)


def _matrix_values(block: str, title: str, min_column: Decimal, source: str) -> list[Decimal]:
    """Values in the columns at or above ``min_column`` of the matrix titled
    ``title``; cells print as "value+SE", and every row has one per column."""
    lines = _section(block, title, "[", source).splitlines()[1:]
    columns = [Decimal(c) for c in lines[0].split()]
    values = []
    for line in lines[1:]:
        if not line.strip() or line.startswith("==="):
            break
        cells = line.split()[1:]
        assert len(cells) == len(columns), f"{source}: row {line!r} under {title!r} is short"
        values += [Decimal(cell.split("+")[0]) for column, cell in zip(columns, cells)
                   if column >= min_column]
    return values


def _crlb_efficiency() -> tuple[str, ...]:
    """Range of the corrected D's spread over sqrt(CRLB) from CNR_0 21, four shapes."""
    path = "experiments/crlb-precision/efficiency_check.txt"
    values = [v for spec in EFFICIENCY_SPECS
              for v in _matrix_values(_section(_read(path), f"=== {spec}:", "=== ", path),
                                      "[all frames] library SD", Decimal(21), path)]
    return _rounded(min(values), 2), _rounded(max(values), 2)


def _per_profile_floors() -> tuple[str, ...]:
    """Per-profile floor at the 0.036 sigma_0^2 tolerance: Gaussian, Voigt, p = 1.5, p = 4."""
    floors = []
    for shape in DFREE_SHAPES:
        path = f"{FLOOR}/per_profile_floor_{shape}.txt"
        floors.append(_rounded(_first(r"tolerance\s+0\.0360 .*CNR ~ ([\d.]+)", _read(path), path),
                               2))
    return tuple(floors)


def _floor_range() -> tuple[str, ...]:
    floors = [Decimal(floor) for floor in _per_profile_floors()]
    return str(min(floors)), str(max(floors))


def _analytical(shape: str) -> tuple[str, ...]:
    """Fisher A, the Fisher / Monte Carlo ratio, and the departure CNR of one shape."""
    path = f"{FLOOR}/analytical_floor.txt"
    block = _section(_read(path), f"=== {shape} ===", "=== ", path)
    return (str(_first(r"A from the Fisher information \(leading_order_terms\): ([\d.]+)",
                       block, path)),
            str(_first(r"ratio Fisher / MC ([\d.]+)", block, path)),
            str(_first(r"differs from 1 by > [\d.]+ SE at CNR ~ ([\d.]+)", block, path)))


def _fisher_ratio_range() -> tuple[str, ...]:
    ratios = [Decimal(_analytical(shape)[1]) for shape in ANALYTICAL_SHAPES]
    return str(min(ratios)), str(max(ratios))


def _fisher_check_ratios() -> tuple[str, ...]:
    """Fisher A over Monte Carlo A per shape (Gaussian, Voigt, p = 1.5, p = 4)."""
    path = "experiments/fisher-calibration/fisher_check.txt"
    return tuple(_search(rf"^\s+{re.escape(spec)}\s+[\d.]+\s+[\d.]+\s+([\d.]+)$",
                         _read(path), path).group(1) for spec in EFFICIENCY_SPECS)


def _fisher_check_range() -> tuple[str, ...]:
    ratios = [Decimal(r) for r in _fisher_check_ratios()]
    return str(min(ratios)), str(max(ratios))


def _subdiffusive_rows() -> list[tuple[str, ...]]:
    """Pooled signed-exponent rows: alpha, then value, SE, z and cell count for
    the naive (1-4), naive + cut (5-8) and corrected (9-12) arms."""
    path = "experiments/subdiffusive-alpha/subdiffusive_alpha.txt"
    text = _section(_read(path), "Signed exponent error", "Mean |exponent", path)
    rows = [(alpha, *_numbers(rest))
            for alpha, rest in re.findall(r"^\s+(\d\.\d\d)\s+\|(.*)$", text, flags=re.M)]
    assert rows and all(len(row) == 13 for row in rows), f"{path}: unexpected exponent rows"
    return rows


def _subdiffusive_row(alpha: str) -> tuple[str, ...]:
    return next(row for row in _subdiffusive_rows() if row[0] == alpha)


def _subdiffusive_corrected() -> tuple[str, ...]:
    return tuple(n for row in _subdiffusive_rows() for n in row[9:11])


def _subdiffusive_naive_range() -> tuple[str, ...]:
    values = [Decimal(row[1]) for row in _subdiffusive_rows()]
    zs = [Decimal(row[3]) for row in _subdiffusive_rows()]
    return str(min(values)), str(max(values)), str(min(zs)), str(max(zs))


def _subdiffusive_corrected_z_bound(alphas: tuple[str, ...]) -> tuple[str, ...]:
    """Largest |z| of the corrected exponent error over the given alphas."""
    return (_bound(max(abs(Decimal(_subdiffusive_row(a)[11])) for a in alphas), 1),)


def _subdiffusive_readme_row(alpha: str) -> tuple[str, ...]:
    """One alpha's naive and corrected value, SE, z and cell count, as the set
    README's table prints them (alpha to one decimal)."""
    row = _subdiffusive_row(alpha)
    return (_rounded(Decimal(alpha), 1), *row[1:5], *row[9:13])


def _multiexp_range() -> tuple[str, ...]:
    path = "experiments/multiexp-decay/multiexp_decay.txt"
    text = _section(_read(path), "(2) Corrected", "(3)", path)
    values = [Decimal(v) for line in re.findall(r"^\s+\d+ \|(.*)$", text, flags=re.M)
              for v in _numbers(line)]
    return _rounded(min(values), 2), _rounded(max(values), 2)


def _joint_weights(column: int) -> tuple[str, ...]:
    """Range of corrected D/D_true with own-width (0) or width-averaged (1) weights."""
    path = f"{DFREE}/joint_weights_check.txt"
    rows = re.findall(r"^\s+[\d.]+\s+\d+\s+([\d.]+) \+/-\s+[\d.]+\s+([\d.]+) \+/-",
                      _read(path), flags=re.M)
    assert rows, f"{path}: no weight rows"
    values = [Decimal(row[column]) for row in rows]
    return _rounded(min(values), 2), _rounded(max(values), 2)


def _calibration_se_ratio() -> tuple[str, ...]:
    """Observed over predicted calibration variance of the cell mean, 16 draws,
    frame 0 weighted (the "current" row's mean)."""
    path = f"{DFREE}/slope_fit_check.txt"
    return (_numbers(_search(r"^\s+current\s+(.*)$", _read(path), path).group(1))[-1],)


def _calibration_se_ratio_18() -> tuple[str, ...]:
    """The same ratio over all 18 draws, the 16 here and two external calibrations."""
    path = f"{DFREE}/calibration_se_check.txt"
    return (str(_first(r"all 18 draws: mean ratio ([\d.]+)", _read(path), path)),)


def _calibration_weight_terms() -> list[Decimal]:
    """The weight term's share of the predicted variance per CNR_0, mean over D."""
    path = f"{DFREE}/calibration_se_check.txt"
    text = _section(_read(path), " Mean over D, per CNR_0", " Three-draw", path)
    return [Decimal(v) for v in _numbers(_search(r"^\s+weights\s+(.*)$", text, path).group(1))]


def _weight_term_from_cnr0_5() -> tuple[str, ...]:
    """Largest weight term from CNR_0 5 up (the columns are CNR_0 2, 3, 5, ...)."""
    return (str(max(_calibration_weight_terms()[2:])),)


def _weight_term_at_cnr0_3() -> tuple[str, ...]:
    return (str(_calibration_weight_terms()[1]),)


def _unbounded_d_row(cnr0: int) -> list[Decimal]:
    """<D_est>/D_true at one CNR_0: bounded OLS, subtract, unbounded OLS, moment."""
    path = f"{ESTIMATOR}/unbounded_fit_comparison.txt"
    text = _section(_read(path), "  <D_est>/D_true", "  SD(D_est", path)
    return [Decimal(v) for v in _numbers(_search(rf"^\s+{cnr0}\s+(.*)$", text, path).group(1))]


def _unbounded_subtract_range() -> tuple[str, ...]:
    values = [_unbounded_d_row(cnr0)[1] for cnr0 in (21, 34, 55, 90)]
    return str(min(values)), str(max(values))


def _unbounded_identical_means() -> tuple[str, ...]:
    """Bounded mean fitted variance where the two fits coincide, at CNR 1.52, 2.00, 2.62."""
    path = f"{ESTIMATOR}/unbounded_fit_comparison.txt"
    return tuple(_rounded(_first(rf"^\s+{re.escape(cnr)}\s+([\d.]+)", _read(path), path), 3)
                 for cnr in ("1.52", "2.00", "2.62"))


def _poc_gaussian_cnr0_5() -> tuple[str, ...]:
    """The Gaussian ``subtract`` mean and combined SE at CNR_0 5."""
    path = "experiments/poc-bias-correction/poc_demo_gaussian.txt"
    text = _section(_read(path), "  <D_est> / D_true", "  mean |bias-1|", path)
    _ols, _inverse_variance, subtract, se = _numbers(_search(r"^\s+5\s+(.*)$", text, path)
                                                     .group(1))
    return _rounded(Decimal(subtract), 2), _rounded(Decimal(se), 2)


def _poc_inverse_variance_cells() -> tuple[str, ...]:
    """Inverse-variance cells beyond 2 SE per shape, then the cells per shape."""
    counts = []
    for shape in POC_SHAPES:
        path = f"experiments/poc-bias-correction/poc_demo_{shape}.txt"
        counts.append(_search(r"inverse-variance cells beyond 2 evaluation SE: (\d+) of (\d+)",
                              _read(path), path).groups())
    assert len({total for _, total in counts}) == 1
    return (*(n for n, _ in counts), counts[0][1])


def _closed_form_chi2_range() -> tuple[str, ...]:
    """chi^2/dof against the exact Fisher value for s >= 10, over the four sigma_0."""
    path = "experiments/closed-form-scaling/linearity_test.txt"
    values = [Decimal(v) for v in re.findall(r"s >= 10: \d+ cells, chi\^2/dof against exact "
                                             r"Fisher ([\d.]+)", _read(path))]
    assert len(values) == 4, f"{path}: expected 4 s >= 10 lines, found {len(values)}"
    return str(min(values)), str(max(values))


def _halide_keys() -> dict[tuple[str, int], dict[str, Decimal]]:
    """D/D_ref per key for each (x, CNR_0) block of the key-sensitivity table."""
    path = f"{HALIDE}/key_sensitivity.txt"
    blocks = {}
    for block in re.split(r"^--- ", _read(path), flags=re.M)[1:]:
        head = re.match(r"x = ([\d.]+), CNR0 = (\d+):", block)
        if head:
            rows = re.findall(r"^\s+(perframe|exact|smoothed|iterated)\s+[-+\d]+\s+[-+\d]+\s+\d+"
                              r"\s+([\d.]+)", block, flags=re.M)
            blocks[(head.group(1), int(head.group(2)))] = {k: Decimal(v) for k, v in rows}
    assert blocks, f"{path}: no key blocks"
    return blocks


def _halide_key_gap(key: str) -> tuple[str, ...]:
    """Largest |D(key) - D(exact)| from CNR_0 15 up, as a "within" bound."""
    gaps = [abs(d[key] - d["exact"]) for (_, cnr0), d in _halide_keys().items() if cnr0 >= 15]
    return (_bound(max(gaps), 3),)


def _halide_perframe_range() -> tuple[str, ...]:
    """Per-frame key D/D_ref at CNR_0 15 and 20 on the two slower-spreading maps."""
    values = [d["perframe"] for (x, cnr0), d in _halide_keys().items()
              if x in ("0.5", "0.75") and cnr0 in (15, 20)]
    return _rounded(min(values), 2), _rounded(max(values), 2)


def _coverage_rows() -> list[dict[str, Decimal]]:
    """The per-cell rows of coverage_check.txt."""
    rows = []
    for line in _read(f"{COVERAGE}/coverage_check.txt").splitlines():
        if re.match(r"^\s+\d+\.\d \|", line):
            (cnr0, min_cnr, _median_sub, median_iv, sd, narrow, _narrow_se, _skew,
             cover_cov, cover_boot, cover_iv, _cover_se) = (Decimal(v) for v in _numbers(line))
            rows.append(dict(cnr0=cnr0, min_cnr=min_cnr, median_iv=median_iv, sd=sd,
                             narrow=narrow, cover_cov=cover_cov, cover_boot=cover_boot,
                             cover_iv=cover_iv))
    assert len(rows) == 13, f"{COVERAGE}/coverage_check.txt: expected 13 cells, found {len(rows)}"
    return rows


def _coverage_range(key: str, rows: list[dict[str, Decimal]] | None = None) -> tuple[str, ...]:
    values = [row[key] for row in (rows if rows is not None else _coverage_rows())]
    return str(min(values)), str(max(values))


def _coverage_pooled(label: str) -> tuple[Decimal, ...]:
    """Numbers of one pooled line of coverage_check.txt, found by its label."""
    path = f"{COVERAGE}/coverage_check.txt"
    line = _search(rf"^.*{re.escape(label)}.*$", _read(path), path).group(0)
    return tuple(Decimal(v) for v in _numbers(line.split(label, 1)[1]))


def _coverage_cells_to_cnr0_21() -> list[dict[str, Decimal]]:
    """Cells the README quotes as "CNR_0 5 to 21" (initial CNR 21.2 and below)."""
    return [row for row in _coverage_rows() if round(row["cnr0"]) <= 21]


def _attribution() -> str:
    return _read(f"{COVERAGE}/narrowness_attribution.txt")


def _attribution_exponent() -> tuple[Decimal, ...]:
    """k, its robust SE and its equal-scatter SE."""
    match = _search(r"k = ([\d.]+) \+/- ([\d.]+) \(robust SE; ([\d.]+) assuming",
                    _attribution(), f"{COVERAGE}/narrowness_attribution.txt")
    return tuple(Decimal(v) for v in match.groups())


def _attribution_sigma3_range() -> tuple[str, ...]:
    values = [Decimal(v) for v in re.findall(r"sigma\^3 scaling of the lookup ([\d.]+)",
                                             _attribution())]
    return str(min(values)), str(max(values))


def _attribution_max_correlation() -> tuple[str, ...]:
    values = re.findall(r"largest \|correlation\| between frames\s+([\d.]+)", _attribution())
    return (str(max(Decimal(v) for v in values)),)


def _attribution_last_width() -> tuple[str, ...]:
    """sigma^2 at the last frame (t = 3 tau), in sigma_0^2."""
    return (_rounded(_first(r"^\s+3\.00\s+([\d.]+)", _attribution(),
                            f"{COVERAGE}/narrowness_attribution.txt"), 2),)


# --------------------------------------------------------------------------
# The claims
# --------------------------------------------------------------------------
ANALYTICAL_ROWS = {
    "gaussian": "| Gaussian | 0.0564 | 0.986 | CNR ≈ 1.18 |",
    "voigt": "| Voigt | 0.0812 | 0.998 | CNR ≈ 0.85 |",
    "generalized_gaussian_p1.5": (
        "| generalized-Gaussian p = 1.5 | 0.1007 | 1.006 | CNR ≈ 1.18 |"),
    "generalized_gaussian_p4": "| generalized-Gaussian p = 4 | 0.0191 | 1.001 | CNR ≈ 1.18 |",
}
FISHER_A_PHRASES = {
    "gaussian": "0.0564 (Gaussian)",
    "voigt": "0.0812 (Voigt)",
    "generalized_gaussian_p1.5": "0.1007 (p = 1.5)",
    "generalized_gaussian_p4": "0.0191 (p = 4)",
}
SUBDIFFUSIVE_README_ROWS = {
    "0.60": "| 0.6 | +0.032 ± 0.009 | +3.8 | 4 | +0.004 ± 0.008 | +0.5 | 4 |",
    "0.80": "| 0.8 | +0.024 ± 0.011 | +2.2 | 4 | −0.011 ± 0.010 | −1.0 | 4 |",
    "1.00": "| 1.0 | +0.065 ± 0.014 | +4.7 | 4 | +0.019 ± 0.014 | +1.4 | 4 |",
    "1.20": "| 1.2 | +0.081 ± 0.016 | +4.9 | 3 | +0.049 ± 0.016 | +3.0 | 4 |",
}
FLOOR_PHRASES = {   # REPO_MAP gives the four floors on one line, so one claim per shape
    0: "CNR 1.16 (Gaussian, Voigt)",
    1: "CNR 1.16 (Gaussian, Voigt)",
    2: "1.51 (p = 1.5)",
    3: "0.89 (p = 4)",
}

CLAIMS = [
    Claim("dfree corrected cells beyond 2 SE",
          lambda: _cells_beyond("standard", "corrected, CNR >= 1"), {
              "README.md": ("4 of 140 cells lie",),
              "docs/DERIVATION.md": ("4 of 140 cells (four shapes",),
              "docs/REPO_MAP.md": ("corrected D: 4 of 140 cells beyond 2 combined SE",),
              "src/width_debias/validate.py": (
                  "4 of 140 cells lie beyond two combined standard errors",)}),
    Claim("dfree corrected cells, every frame",
          lambda: _cells_beyond("standard", "corrected, all frames"), {
              "src/width_debias/validate.py": ("5 of 140 with every frame",)}),
    Claim("dfree naive cells", lambda: _cells_beyond("standard", "naive, all frames"), {
        "README.md": ("has 46 of 140",),
        "docs/DERIVATION.md": ("46 of 140 cells beyond 2 SE",),
        "docs/REPO_MAP.md": ("naive all frames 46 of 140",)}),
    Claim("dfree short-window corrected cells",
          lambda: _cells_beyond("short", "corrected, CNR >= 1"), {
              "docs/DERIVATION.md": ("5 of 96 corrected",),
              "docs/REPO_MAP.md": ("5 of 96 (short window)",),
              f"{DFREE}/README.md": ("p = 4; 5 of 96)",)}),
    Claim("dfree short-window naive cells", lambda: _cells_beyond("short", "naive, all frames"), {
        "docs/DERIVATION.md": ("41 of 96 uncorrected",),
        "docs/REPO_MAP.md": ("46 of 140 and 41 of 96",),
        f"{DFREE}/README.md": ("(41 of 96)",)}),
    Claim("dfree pooled corrected bias bound", _dfree_pooled_bound, {
        "README.md": ("within 0.4 combined standard errors",),
        "docs/DERIVATION.md": ("within 0.4 combined SE of zero",),
        "docs/REPO_MAP.md": ("pooled bias within 0.4 combined SE",)}),
    Claim("dense subtract pooled bias bound", _dense_pooled_bound, {
        "docs/DERIVATION.md": ("within 0.6 combined SE of zero on every shape",),
        "docs/REPO_MAP.md": ("dense sweep pooled bias within 0.6 combined SE",)}),
    Claim("calibration SE, observed over predicted variance", _calibration_se_ratio, {
        "docs/DERIVATION.md": ("variance of the cell mean 0.89 with the per-frame slope weights",),
        "docs/REPO_MAP.md": ("observed/predicted 0.89 over 16 draws",),
        "src/width_debias/weights.py": ("of the cell mean 0.89 over the 16 draws",)}),
    Claim("calibration SE, observed over predicted variance, 18 draws",
          _calibration_se_ratio_18, {
              "src/width_debias/weights.py": ("and 0.93 with two external calibrations",)}),
    Claim("calibration SE, weight term from CNR_0 5", _weight_term_from_cnr0_5, {
        f"{DFREE}/README.md": ("term is at most 0.04 of the prediction",),
        "src/width_debias/weights.py": ("the weight term is at most 0.04 of it",)}),
    Claim("calibration SE, weight term at CNR_0 3", _weight_term_at_cnr0_3, {
        f"{DFREE}/README.md": ("and 0.22 at CNR₀ 2 and 3)",),
        "src/width_debias/weights.py": ("and 0.22 of it",)}),
    Claim("CRLB efficiency from CNR_0 21", _crlb_efficiency, {
        "docs/DERIVATION.md": ("0.95 to 1.10 × √CRLB",),
        "docs/REPO_MAP.md": ("reaches 0.95-1.10 x sqrt(CRLB) from CNR0 21",),
        "experiments/crlb-precision/README.md": ("0.95 to 1.10 x sqrt(CRLB)",),
        "experiments/crlb-precision/precision_limit.md": ("0.95 to 1.10 × √CRLB",),
        "src/width_debias/weights.py": ("0.95 to 1.10 x sqrt(CRLB)",)}),
    Claim("per-profile floors at 0.036 sigma_0^2", _per_profile_floors, {
        "docs/DERIVATION.md": (
            "| 0.036 | 0.3 × 2 D t_max, D = 0.02 | 1.16 | 1.16 | 1.51 | 0.89 |",)}),
    *[Claim(f"per-profile floor, {DFREE_SHAPES[i]}", lambda i=i: (_per_profile_floors()[i],), {
        "docs/REPO_MAP.md": (phrase,)}) for i, phrase in FLOOR_PHRASES.items()],
    Claim("per-profile floor range", _floor_range, {
        "README.md": ("(0.89 to 1.51 depending on shape)", "(0.89 to 1.51 depending on shape:")}),
    *[Claim(f"analytical floor, {shape}", lambda shape=shape: _analytical(shape), {
        "docs/DERIVATION.md": (ANALYTICAL_ROWS[shape],)}) for shape in ANALYTICAL_SHAPES],
    *[Claim(f"Fisher A, {shape}", lambda shape=shape: _analytical(shape)[:1], {
        f"{FLOOR}/README.md": (FISHER_A_PHRASES[shape],)}) for shape in ANALYTICAL_SHAPES],
    Claim("Fisher / Monte Carlo ratio range", _fisher_ratio_range, {
        "docs/DERIVATION.md": ("reproduces the Fisher `A` to 0.986 to 1.006",),
        "docs/REPO_MAP.md": ("Fisher A matches Monte Carlo to 0.986-1.006",),
        f"{FLOOR}/README.md": ("of 0.986 to 1.006",)}),
    Claim("leading-order departure, Gaussian and Voigt",
          lambda: (_analytical("gaussian")[2], _analytical("voigt")[2]), {
              "docs/REPO_MAP.md": ("departs at CNR about 1.18 (0.85 for Voigt)",)}),
    Claim("Fisher check, ratio per shape", _fisher_check_ratios, {
        "experiments/fisher-calibration/README.md": ("ratio 0.999, 1.003, 0.998, 0.998",)}),
    Claim("Fisher check, ratio range", _fisher_check_range, {
        "docs/REPO_MAP.md": ("Fisher A matches Monte Carlo to 0.998-1.003",)}),
    Claim("closed-form scaling, chi^2/dof from s = 10", _closed_form_chi2_range, {
        "docs/REPO_MAP.md": ("(chi^2/dof 0.88-0.97 at four sigma0)",)}),
    Claim("subdiffusive corrected exponent errors", _subdiffusive_corrected, {
        "docs/DERIVATION.md": (
            "+0.004 ± 0.008, −0.011 ± 0.010, +0.019 ± 0.014, and +0.049 ± 0.016",)}),
    Claim("subdiffusive naive exponent errors", _subdiffusive_naive_range, {
        "docs/DERIVATION.md": ("+0.024 to +0.081 naive (z 2.2 to 4.9)",),
        "docs/REPO_MAP.md": ("naive +0.024 to +0.081 (z 2.2-4.9)",)}),
    Claim("subdiffusive alpha 1.2 corrected", lambda: _subdiffusive_row("1.20")[9:11], {
        "docs/REPO_MAP.md": ("+0.049 ± 0.016 for 1.2",)}),
    Claim("subdiffusive corrected z, alpha 0.6 to 1.0",
          lambda: _subdiffusive_corrected_z_bound(("0.60", "0.80", "1.00")), {
              "docs/REPO_MAP.md": ("within 1.4 SE for alpha 0.6-1.0",)}),
    *[Claim(f"subdiffusive table row, alpha {alpha}",
            lambda alpha=alpha: _subdiffusive_readme_row(alpha), {
                "experiments/subdiffusive-alpha/README.md": (row,)})
      for alpha, row in SUBDIFFUSIVE_README_ROWS.items()],
    Claim("multi-exponential corrected D range", _multiexp_range, {
        "docs/DERIVATION.md": ("within 0.93 to 1.06 of D_ref",),
        "docs/REPO_MAP.md": ("(corrected D 0.93-1.06 of D_ref",),
        "experiments/multiexp-decay/README.md": ("0.93 to 1.06 over CNR₀ 8 to 90",)}),
    Claim("joint lookup, own-width weights", lambda: _joint_weights(0), {
        "docs/REPO_MAP.md": ("against 0.39 to 0.96 read at each frame's own fitted width",),
        f"{DFREE}/README.md": ("the corrected D/D_true is 0.39 to 0.96",),
        "src/width_debias/weights.py": ("D/D_true at 0.39 to 0.96",)}),
    Claim("joint lookup, width-averaged weights", lambda: _joint_weights(1), {
        "docs/REPO_MAP.md": ("corrected D/D_true 0.88 to 1.09",),
        f"{DFREE}/README.md": ("width-averaged weights it is 0.88 to 1.09",),
        "src/width_debias/weights.py": ("against 0.88 to 1.09 for the",)}),
    Claim("high-N halide, smoothed key against the exact key from CNR_0 15",
          lambda: _halide_key_gap("smoothed"), {
              f"{HALIDE}/README.md": ("the smoothed key gives the same D as the exact key to "
                                      "within 0.004",),
              "docs/DERIVATION.md": ("matches the exact key to within 0.004 from CNR₀ 15",),
              "src/width_debias/bias_lookup.py": (
                  "matched the exact key to within 0.004 from CNR_0 15",)}),
    Claim("high-N halide, iterated key against the exact key from CNR_0 15",
          lambda: _halide_key_gap("iterated"), {
              f"{HALIDE}/README.md": ("and the iterated key to within 0.002",)}),
    Claim("high-N halide, per-frame key at CNR_0 15 and 20", _halide_perframe_range, {
        "docs/DERIVATION.md": ("leaves D at 0.89 to 0.96 of the reference",),
        "src/width_debias/bias_lookup.py": ("left D at 0.89 to 0.96 of the reference",)}),
    Claim("unbounded comparison, OLS at CNR_0 34",
          lambda: (_rounded(_unbounded_d_row(34)[0], 2),), {
              "README.md": ("an unweighted fit returns 1.20 times the reference D",)}),
    Claim("unbounded comparison, subtract at CNR_0 34",
          lambda: (_rounded(_unbounded_d_row(34)[1], 2),), {
              "README.md": ("the corrected fit returns 1.01 times the reference",)}),
    Claim("unbounded comparison, subtract from CNR_0 21", _unbounded_subtract_range, {
        f"{ESTIMATOR}/README.md": ("reads 1.003 to 1.028",),
        f"{ESTIMATOR}/unbounded_fit_comparison_caption.md": ("reads 1.003 to 1.028",)}),
    Claim("bounded and unbounded fits coincide", _unbounded_identical_means, {
        f"{ESTIMATOR}/README.md": ("fitted variance 1.038, 1.016, 1.010 sigma_0^2",),
        f"{ESTIMATOR}/unbounded_fit_comparison_caption.md": ("(1.038, 1.016, and 1.010 σ₀²",),
        "src/width_debias/models.py": ("variance 1.038, 1.016, 1.010 sigma_0^2 at CNR 1.52",)}),
    Claim("proof of concept, Gaussian CNR_0 5", _poc_gaussian_cnr0_5, {
        "experiments/poc-bias-correction/README.md": ("2.67 ± 0.86",),
        "experiments/poc-bias-correction/poc_demo_caption.md": ("2.67 ± 0.86 at CNR₀ 5",)}),
    Claim("proof of concept, inverse-variance cells beyond 2 SE", _poc_inverse_variance_cells, {
        "docs/REPO_MAP.md": ("inverse-variance leaves 1, 2, 4, and 0 of 7 cells beyond 2 SE",)}),
    Claim("coverage, covariance interval per cell", lambda: _coverage_range("cover_cov"), {
        f"{COVERAGE}/README.md": ("covers D_ref 0.939 to 0.952 of the",)}),
    Claim("coverage, covariance interval pooled", lambda: tuple(
        str(v) for v in _coverage_pooled("over 13 cells, covariance, subtract:")), {
        f"{COVERAGE}/README.md": ("0.9438 ± 0.0007 pooled over the 13 cells",)}),
    Claim("coverage, covariance interval pooled, rounded", lambda: (
        _rounded(_coverage_pooled("over 13 cells, covariance, subtract:")[0], 3),), {
        f"{COVERAGE}/coverage_check_caption.md": ("covariance interval covers 0.944 and",),
        "docs/REPO_MAP.md": ("covariance interval covers 0.944 pooled over 13 CNR0",)}),
    Claim("coverage, narrowness where every frame has CNR >= 1", lambda: tuple(
        _rounded(v, 3) for v in _coverage_pooled("narrowness, plain mean:")), {
        f"{COVERAGE}/README.md": ("D_se is 0.967 ± 0.003 of the realised",),
        f"{COVERAGE}/coverage_check_caption.md": (
            "it is 0.967, with a cell-to-cell SE of 0.003",)}),
    Claim("coverage, narrowness where every frame has CNR >= 1, value", lambda: (
        _rounded(_coverage_pooled("narrowness, plain mean:")[0], 3),), {
        "docs/REPO_MAP.md": ("its SE is 0.967 of the realised spread",)}),
    Claim("coverage, narrowness per cell", lambda: _coverage_range("narrow"), {
        f"{COVERAGE}/README.md": ("(0.944 to 1.003 per cell over the whole sweep)",)}),
    Claim("coverage, implied and measured where every frame has CNR >= 1", lambda: (
        _rounded(_coverage_pooled("implied by the narrowness for a normal D_hat:")[0], 3),
        *(_rounded(v, 3) for v in _coverage_pooled("covariance coverage, subtract:")[:2])), {
        f"{COVERAGE}/README.md": ("implies a coverage of 0.942, and the covariance interval "
                                  "covers 0.942 ± 0.001",)}),
    Claim("coverage, bootstrap per cell and pooled", lambda: (
        *_coverage_range("cover_boot"),
        *(str(v) for v in _coverage_pooled("over 13 cells, bootstrap, subtract:"))), {
        f"{COVERAGE}/README.md": ("covers 0.945 to 0.955, 0.9505 ± 0.0007 pooled",)}),
    Claim("coverage, bootstrap pooled, rounded", lambda: (
        _rounded(_coverage_pooled("over 13 cells, bootstrap, subtract:")[0], 3),), {
        f"{COVERAGE}/coverage_check_caption.md": ("the bootstrap 0.951",),
        "docs/REPO_MAP.md": ("per-frame variances covers 0.951",)}),
    Claim("coverage, inverse-variance contrast per cell", lambda: _coverage_range("cover_iv"), {
        f"{COVERAGE}/README.md": ("inverse-variance contrast covers 0.936 to 0.953",)}),
    Claim("coverage, inverse-variance median to CNR_0 21",
          lambda: _coverage_range("median_iv", _coverage_cells_to_cnr0_21()), {
              f"{COVERAGE}/README.md": ("(median 1.05 to 1.57 of D_ref",)}),
    Claim("coverage, single-measurement SD to CNR_0 21",
          lambda: _coverage_range("sd", _coverage_cells_to_cnr0_21()), {
              f"{COVERAGE}/README.md": ("(SD 0.82 to 4.61 of D_ref)",)}),
    Claim("attribution, exponent k and its robust SE",
          lambda: tuple(str(v) for v in _attribution_exponent()[:2]), {
              f"{COVERAGE}/README.md": ("k = 1.36 ± 0.18",)}),
    Claim("attribution, equal-scatter SE of k", lambda: (str(_attribution_exponent()[2]),), {
        f"{COVERAGE}/README.md": ("0.13 assuming equal scatter",)}),
    Claim("attribution, sigma^3 prediction", _attribution_sigma3_range, {
        f"{COVERAGE}/README.md": ("D_se / SD(D̂) of 0.964 to 0.971",)}),
    Claim("attribution, largest correlation between frames", _attribution_max_correlation, {
        f"{COVERAGE}/README.md": ("correlation between frames is at most 0.052",)}),
    Claim("attribution, width at the last frame", _attribution_last_width, {
        f"{COVERAGE}/README.md": ("the width grows to σ² = 1.12 σ₀²",),
        f"{COVERAGE}/coverage_check_caption.md": (
            "width grows to 1.12 σ₀² by the last frame",)}),
]

# Superseded values, allowed only in the documents that keep history.
RETIRED = {
    "0.98 to 1.08": HISTORY_DOCUMENTS,
    "136 of 140": HISTORY_DOCUMENTS,
    "4 of 96": HISTORY_DOCUMENTS,
    "1.44 to 14)": HISTORY_DOCUMENTS,
    "0.089 vs": HISTORY_DOCUMENTS,
    "1.00 times the reference": HISTORY_DOCUMENTS,
    "0.98 over 18 draws": HISTORY_DOCUMENTS,
}


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def _copies() -> list:
    return [pytest.param(claim, path, snippet, id=f"{claim.name} | {path}")
            for claim in CLAIMS
            for path, snippets in claim.path_to_snippets.items() for snippet in snippets]


def _carries(snippet_numbers: tuple[str, ...], value: tuple[str, ...]) -> bool:
    width = len(value)
    return any(snippet_numbers[i:i + width] == value
               for i in range(len(snippet_numbers) - width + 1))


@pytest.mark.parametrize("claim, path, snippet", _copies())
def test_quoted_number_matches_its_table(claim: Claim, path: str, snippet: str) -> None:
    """The copy is in its document and carries the value its table gives now."""
    value = claim.compute()
    assert _contains(_read(path), snippet), (
        f"{path} no longer contains {snippet!r} for {claim.name!r}: update the document "
        "and this snippet together")
    assert _carries(_numbers(snippet), value), (
        f"{path} quotes {snippet!r}, but the table now gives {', '.join(value)} for "
        f"{claim.name!r}: update the copy in {path} and this snippet")


def _tracked_documents() -> list[str]:
    """Tracked Markdown documents and library modules (what CI checks out)."""
    try:
        listing = subprocess.run(["git", "ls-files", "--", "*.md", "src/*.py"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout to list the tracked documents")
    return sorted(path for path in listing.splitlines() if path)


def _range_dashes_as_to(text: str) -> str:
    """Read a dash between two numbers as "to", so "0.98-1.08" and "0.98 to 1.08" agree."""
    return re.sub(r"(?<=\d)\s*[-–—]\s*(?=\d)", " to ", text)


@pytest.mark.parametrize("retired", sorted(RETIRED))
def test_retired_value_does_not_reappear(retired: str) -> None:
    """A superseded number stays only in the documents that keep history."""
    value = _range_dashes_as_to(retired)
    pattern = re.compile(NOT_AFTER_NUMBER + re.escape(value)
                         + (NOT_BEFORE_NUMBER if value[-1].isdigit() else ""))
    found = [path for path in _tracked_documents() if path not in RETIRED[retired]
             and pattern.search(_range_dashes_as_to(_flat(_read(path))))]
    assert not found, f"superseded {retired!r} still quoted in {found}"
