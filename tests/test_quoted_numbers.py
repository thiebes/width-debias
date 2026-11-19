"""Numbers quoted in the documentation match the committed tables they come from.

The validation numbers are restated by hand in the README, docs/DERIVATION.md,
docs/REPO_MAP.md, set READMEs, captions and library docstrings, and a rerun
that changes a table leaves the copies behind. Each claim below recomputes a
quoted value from the committed ``.txt`` table that produced it and checks that
every listed copy still carries it. When a rerun changes a table, the claim
fails until each copy is updated; when a copy is reworded, its snippet fails
until the claim is. ``RETIRED`` lists superseded values that must not
reappear outside the documents that keep history.

A copy matches when its snippet appears in the document (whitespace runs
collapsed, so line breaks and docstring indentation do not matter) and the
claim's numbers appear in the snippet in order and side by side. Numbers are
compared as printed, with a leading plus sign dropped and the Unicode minus
read as a hyphen, so "+0.004 ± 0.008" and "0.004 +/- 0.008" both carry
("0.004", "0.008").

The coverage-interval set is not covered yet: its text is under revision.
"""
from __future__ import annotations

import re
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
DFREE_SHAPES = ("gaussian", "voigt", "genGauss_p1.5", "genGauss_p4")
ANALYTICAL_SHAPES = ("gaussian", "voigt", "generalized_gaussian_p1.5", "generalized_gaussian_p4")
EFFICIENCY_SPECS = ("gaussian", "voigt", "generalized_gaussian:1.5", "generalized_gaussian:4")
HISTORY_DOCUMENTS = frozenset({"CHANGELOG.md", "CLAUDE.md", "PLAN.md", "docs/STATUS.md"})
SKIPPED_DIRECTORIES = {".git", ".venv", "archive", "build", "output", "node_modules"}

# A number not glued to a preceding word or digit: in "0.6-1.0" the hyphen is a
# range dash, not a sign, and "sigma_0" carries no number.
NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Claim:
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
    return " ".join(text.split())


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(m.replace("−", "-").lstrip("+") for m in NUMBER.findall(text))


def _rounded(value: Decimal, places: int) -> str:
    return str(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _bound(value: Decimal, places: int) -> str:
    """The smallest number at ``places`` decimals not below ``value``, as a
    "within X" statement quotes it."""
    return str(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_CEILING))


def _section(text: str, start: str, next_start: str) -> str:
    """From the line beginning ``start`` to the next line beginning ``next_start``."""
    begin = text.index(start)
    end = text.find("\n" + next_start, begin + len(start))
    return text[begin:] if end < 0 else text[begin:end]


def _first(pattern: str, text: str) -> Decimal:
    return Decimal(re.search(pattern, text, flags=re.M).group(1))


# --------------------------------------------------------------------------
# Table readers, one per kind of claim
# --------------------------------------------------------------------------
def _dfree_arm(shape: str, window: str, arm: str) -> str:
    stem = "validate_d_zero" if window == "standard" else "validate_d_zero_short"
    return _section(_read(f"{DFREE}/{stem}_{shape}.txt"), f"--- {arm}\n", "--- ")


def _cells_beyond(window: str, arm: str) -> tuple[str, ...]:
    """Cells beyond two combined SE, summed over the four shapes, as N and M of "N of M"."""
    counts = [re.search(r"cells beyond 2 combined SE: (\d+) of (\d+)",
                        _dfree_arm(shape, window, arm)).groups() for shape in DFREE_SHAPES]
    return str(sum(int(n) for n, _ in counts)), str(sum(int(m) for _, m in counts))


def _dfree_pooled_bound() -> tuple[str, ...]:
    """Largest |pooled z| of the corrected arm (CNR >= 1) over the four shapes."""
    z = max(abs(_first(r"all cells: .*; ([-+]?[\d.]+) combined SE",
                       _dfree_arm(shape, "standard", "corrected, CNR >= 1")))
            for shape in DFREE_SHAPES)
    return (_bound(z, 1),)


def _dense_pooled_bound() -> tuple[str, ...]:
    """Largest |pooled z| of ``subtract`` over all test points of the four shapes."""
    text = _read(f"{DFREE}/dense_calibration_and_test_four_shapes.txt")
    z = max(abs(Decimal(v)) for v in re.findall(
        r"subtract: pooled.*?all points .*?; ([-+]?[\d.]+) SE", text, flags=re.S))
    return (_bound(z, 1),)


def _matrix_values(block: str, title: str, min_column: Decimal) -> list[Decimal]:
    """Values in the columns at or above ``min_column`` of the matrix titled
    ``title``; cells print as "value+SE"."""
    lines = block[block.index(title):].splitlines()[1:]
    columns = [Decimal(c) for c in lines[0].split()]
    values = []
    for line in lines[1:]:
        if not line.strip() or line.startswith(("[", "===")):
            break
        values += [Decimal(cell.split("+")[0]) for column, cell in zip(columns, line.split()[1:])
                   if column >= min_column]
    return values


def _crlb_efficiency() -> tuple[str, ...]:
    """Range of the corrected D's spread over sqrt(CRLB) from CNR_0 21, four shapes."""
    text = _read("experiments/crlb-precision/efficiency_check.txt")
    values = [v for spec in EFFICIENCY_SPECS
              for v in _matrix_values(_section(text, f"=== {spec}:", "=== "),
                                      "[all frames] library SD", Decimal(21))]
    return _rounded(min(values), 2), _rounded(max(values), 2)


def _per_profile_floors() -> tuple[str, ...]:
    """Per-profile floor at the 0.036 sigma_0^2 tolerance: Gaussian, Voigt, p = 1.5, p = 4."""
    return tuple(_rounded(_first(r"tolerance\s+0\.0360 .*CNR ~ ([\d.]+)",
                                 _read(f"{FLOOR}/per_profile_floor_{shape}.txt")), 2)
                 for shape in DFREE_SHAPES)


def _floor_range() -> tuple[str, ...]:
    floors = [Decimal(floor) for floor in _per_profile_floors()]
    return str(min(floors)), str(max(floors))


def _analytical(shape: str) -> tuple[str, ...]:
    """Fisher A, the Fisher / Monte Carlo ratio, and the departure CNR of one shape."""
    block = _section(_read(f"{FLOOR}/analytical_floor.txt"), f"=== {shape} ===", "=== ")
    return (str(_first(r"A from the Fisher information \(leading_order_terms\): ([\d.]+)", block)),
            str(_first(r"ratio Fisher / MC ([\d.]+)", block)),
            str(_first(r"differs from 1 by > [\d.]+ SE at CNR ~ ([\d.]+)", block)))


def _fisher_ratio_range() -> tuple[str, ...]:
    ratios = [Decimal(_analytical(shape)[1]) for shape in ANALYTICAL_SHAPES]
    return str(min(ratios)), str(max(ratios))


def _subdiffusive_rows() -> list[tuple[str, ...]]:
    """Pooled signed-exponent rows: alpha, then value, SE, z and cell count for
    the naive (1-4), naive + cut (5-8) and corrected (9-12) arms."""
    text = _section(_read("experiments/subdiffusive-alpha/subdiffusive_alpha.txt"),
                    "Signed exponent error", "Mean |exponent")
    return [(alpha, *_numbers(rest))
            for alpha, rest in re.findall(r"^\s+(\d\.\d\d)\s+\|(.*)$", text, flags=re.M)]


def _subdiffusive_row(alpha: str) -> tuple[str, ...]:
    return next(row for row in _subdiffusive_rows() if row[0] == alpha)


def _subdiffusive_corrected() -> tuple[str, ...]:
    return tuple(n for row in _subdiffusive_rows() for n in row[9:11])


def _subdiffusive_naive_range() -> tuple[str, ...]:
    values = [Decimal(row[1]) for row in _subdiffusive_rows()]
    zs = [Decimal(row[3]) for row in _subdiffusive_rows()]
    return str(min(values)), str(max(values)), str(min(zs)), str(max(zs))


def _subdiffusive_readme_row(alpha: str) -> tuple[str, ...]:
    """One alpha's naive and corrected value, SE, z and cell count, as the set
    README's table prints them (alpha to one decimal)."""
    row = _subdiffusive_row(alpha)
    return (_rounded(Decimal(alpha), 1), *row[1:5], *row[9:13])


def _multiexp_range() -> tuple[str, ...]:
    text = _section(_read("experiments/multiexp-decay/multiexp_decay.txt"), "(2) Corrected", "(3)")
    values = [Decimal(v) for line in re.findall(r"^\s+\d+ \|(.*)$", text, flags=re.M)
              for v in _numbers(line)]
    return _rounded(min(values), 2), _rounded(max(values), 2)


def _joint_weights(column: int) -> tuple[str, ...]:
    """Range of corrected D/D_true with own-width (0) or width-averaged (1) weights."""
    rows = re.findall(r"^\s+[\d.]+\s+\d+\s+([\d.]+) \+/-\s+[\d.]+\s+([\d.]+) \+/-",
                      _read(f"{DFREE}/joint_weights_check.txt"), flags=re.M)
    values = [Decimal(row[column]) for row in rows]
    return _rounded(min(values), 2), _rounded(max(values), 2)


def _unbounded_d_row(cnr0: int) -> list[Decimal]:
    """<D_est>/D_true at one CNR_0: bounded OLS, subtract, unbounded OLS, moment."""
    text = _section(_read(f"{ESTIMATOR}/unbounded_fit_comparison.txt"),
                    "  <D_est>/D_true", "  SD(D_est")
    line = re.search(rf"^\s+{cnr0}\s+(.*)$", text, flags=re.M).group(1)
    return [Decimal(v) for v in _numbers(line)]


def _unbounded_subtract_range() -> tuple[str, ...]:
    values = [_unbounded_d_row(cnr0)[1] for cnr0 in (21, 34, 55, 90)]
    return str(min(values)), str(max(values))


def _unbounded_identical_means() -> tuple[str, ...]:
    """Bounded mean fitted variance where the two fits coincide, at CNR 1.52, 2.00, 2.62."""
    text = _read(f"{ESTIMATOR}/unbounded_fit_comparison.txt")
    return tuple(_rounded(_first(rf"^\s+{re.escape(cnr)}\s+([\d.]+)", text), 3)
                 for cnr in ("1.52", "2.00", "2.62"))


def _poc_gaussian_cnr0_5() -> tuple[str, ...]:
    """The Gaussian ``subtract`` mean and combined SE at CNR_0 5."""
    text = _section(_read("experiments/poc-bias-correction/poc_demo_gaussian.txt"),
                    "  <D_est> / D_true", "  mean |bias-1|")
    _ols, _inverse_variance, subtract, se = _numbers(re.search(r"^\s+5\s+(.*)$", text, flags=re.M).group(1))
    return _rounded(Decimal(subtract), 2), _rounded(Decimal(se), 2)


# --------------------------------------------------------------------------
# The claims
# --------------------------------------------------------------------------
ANALYTICAL_ROWS = {
    "gaussian": "| Gaussian | 0.0564 | 0.986 | CNR ≈ 1.18 |",
    "voigt": "| Voigt | 0.0812 | 0.998 | CNR ≈ 0.85 |",
    "generalized_gaussian_p1.5": "| generalized-Gaussian p = 1.5 | 0.1007 | 1.006 | CNR ≈ 1.18 |",
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

CLAIMS = [
    Claim("dfree corrected cells beyond 2 SE", lambda: _cells_beyond("standard", "corrected, CNR >= 1"), {
        "README.md": ("4 of 140 cells lie",),
        "docs/DERIVATION.md": ("4 of 140 cells (four shapes",),
        "docs/REPO_MAP.md": ("corrected D: 4 of 140 cells beyond 2 combined SE",),
        "src/width_debias/validate.py": ("4 of 140 cells lie beyond two combined standard errors",)}),
    Claim("dfree corrected cells, every frame", lambda: _cells_beyond("standard", "corrected, all frames"), {
        "src/width_debias/validate.py": ("5 of 140 with every frame",)}),
    Claim("dfree naive cells", lambda: _cells_beyond("standard", "naive, all frames"), {
        "README.md": ("has 46 of 140",),
        "docs/DERIVATION.md": ("46 of 140 cells beyond 2 SE",),
        "docs/REPO_MAP.md": ("naive all frames 46 of 140",)}),
    Claim("dfree short-window corrected cells", lambda: _cells_beyond("short", "corrected, CNR >= 1"), {
        "docs/DERIVATION.md": ("5 of 96 corrected",),
        "docs/REPO_MAP.md": ("5 of 96 (short window)",)}),
    Claim("dfree short-window naive cells", lambda: _cells_beyond("short", "naive, all frames"), {
        "docs/DERIVATION.md": ("41 of 96 uncorrected",),
        "docs/REPO_MAP.md": ("41 of 96",),
        "experiments/dfree-calibration/README.md": ("(41 of 96)",)}),
    Claim("dfree pooled corrected bias bound", _dfree_pooled_bound, {
        "README.md": ("within 0.4 combined standard errors",),
        "docs/DERIVATION.md": ("within 0.4 combined SE of zero",),
        "docs/REPO_MAP.md": ("pooled bias within 0.4 combined SE",)}),
    Claim("dense subtract pooled bias bound", _dense_pooled_bound, {
        "docs/DERIVATION.md": ("within 0.6 combined SE of zero on every shape",),
        "docs/REPO_MAP.md": ("dense sweep pooled bias within 0.6 combined SE",)}),
    Claim("CRLB efficiency from CNR_0 21", _crlb_efficiency, {
        "docs/DERIVATION.md": ("0.95 to 1.10 × √CRLB",),
        "experiments/crlb-precision/README.md": ("0.95 to 1.10 x sqrt(CRLB)",),
        "experiments/crlb-precision/precision_limit.md": ("0.95 to 1.10 × √CRLB",),
        "src/width_debias/weights.py": ("0.95 to 1.10 x sqrt(CRLB)",)}),
    Claim("per-profile floors at 0.036 sigma_0^2", _per_profile_floors, {
        "docs/DERIVATION.md": ("| 0.036 | 0.3 × 2 D t_max, D = 0.02 | 1.16 | 1.16 | 1.51 | 0.89 |",)}),
    Claim("per-profile floor range", _floor_range, {
        "README.md": ("(0.89 to 1.51 depending on shape)", "(0.89 to 1.51 depending on shape:")}),
    *[Claim(f"analytical floor, {shape}", lambda shape=shape: _analytical(shape), {
        "docs/DERIVATION.md": (ANALYTICAL_ROWS[shape],)}) for shape in ANALYTICAL_SHAPES],
    *[Claim(f"Fisher A, {shape}", lambda shape=shape: _analytical(shape)[:1], {
        "experiments/correction-floor/README.md": (FISHER_A_PHRASES[shape],)}) for shape in ANALYTICAL_SHAPES],
    Claim("Fisher / Monte Carlo ratio range", _fisher_ratio_range, {
        "experiments/correction-floor/README.md": ("of 0.986 to 1.006",)}),
    Claim("subdiffusive corrected exponent errors", _subdiffusive_corrected, {
        "docs/DERIVATION.md": ("+0.004 ± 0.008, −0.011 ± 0.010, +0.019 ± 0.014, and +0.049 ± 0.016",)}),
    Claim("subdiffusive naive exponent errors", _subdiffusive_naive_range, {
        "docs/DERIVATION.md": ("+0.024 to +0.081 naive (z 2.2 to 4.9)",),
        "docs/REPO_MAP.md": ("naive +0.024 to +0.081 (z 2.2-4.9)",)}),
    Claim("subdiffusive alpha 1.2 corrected", lambda: _subdiffusive_row("1.20")[9:11], {
        "docs/REPO_MAP.md": ("+0.049 ± 0.016 for 1.2",)}),
    *[Claim(f"subdiffusive table row, alpha {alpha}", lambda alpha=alpha: _subdiffusive_readme_row(alpha), {
        "experiments/subdiffusive-alpha/README.md": (row,)}) for alpha, row in SUBDIFFUSIVE_README_ROWS.items()],
    Claim("multi-exponential corrected D range", _multiexp_range, {
        "docs/DERIVATION.md": ("within 0.93 to 1.06 of D_ref",),
        "experiments/multiexp-decay/README.md": ("0.93 to 1.06 over CNR₀ 8 to 90",)}),
    Claim("joint lookup, own-width weights", lambda: _joint_weights(0), {
        "docs/REPO_MAP.md": ("against 0.39 to 0.96 read at each frame's own fitted width",),
        "experiments/dfree-calibration/README.md": ("the corrected D/D_true is 0.39 to 0.96",),
        "src/width_debias/weights.py": ("D/D_true at 0.39 to 0.96",)}),
    Claim("joint lookup, width-averaged weights", lambda: _joint_weights(1), {
        "docs/REPO_MAP.md": ("corrected D/D_true 0.88 to 1.09",),
        "experiments/dfree-calibration/README.md": ("width-averaged weights it is 0.88 to 1.09",),
        "src/width_debias/weights.py": ("against 0.88 to 1.09 for the",)}),
    Claim("unbounded comparison, OLS at CNR_0 34", lambda: (_rounded(_unbounded_d_row(34)[0], 2),), {
        "README.md": ("an unweighted fit returns 1.20 times the reference D",)}),
    Claim("unbounded comparison, subtract at CNR_0 34", lambda: (_rounded(_unbounded_d_row(34)[1], 2),), {
        "README.md": ("the corrected fit returns 1.01 times the reference",)}),
    Claim("unbounded comparison, subtract from CNR_0 21", _unbounded_subtract_range, {
        f"{ESTIMATOR}/README.md": ("reads 1.003 to 1.028",),
        f"{ESTIMATOR}/unbounded_fit_comparison_caption.md": ("reads 1.003 to 1.028",)}),
    Claim("bounded and unbounded fits coincide", _unbounded_identical_means, {
        f"{ESTIMATOR}/README.md": ("fitted variance 1.038, 1.016, 1.010 sigma_0^2",),
        f"{ESTIMATOR}/unbounded_fit_comparison_caption.md": ("(1.038, 1.016, and 1.010 σ₀²",)}),
    Claim("proof of concept, Gaussian CNR_0 5", _poc_gaussian_cnr0_5, {
        "experiments/poc-bias-correction/README.md": ("2.67 ± 0.86",),
        "experiments/poc-bias-correction/poc_demo_caption.md": ("2.67 ± 0.86 at CNR₀ 5",)}),
]

# Superseded values and the documents allowed to keep them as history.
RETIRED = {
    "0.98 to 1.08": HISTORY_DOCUMENTS,
    "136 of 140": HISTORY_DOCUMENTS,
    "4 of 96": HISTORY_DOCUMENTS,
    "1.44 to 14)": HISTORY_DOCUMENTS,
    "0.089 vs": HISTORY_DOCUMENTS,
    "1.00 times the reference": HISTORY_DOCUMENTS,
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
    assert _flat(snippet) in _flat(_read(path)), (
        f"{path} no longer contains {snippet!r}; update the claim {claim.name!r}")
    assert _carries(_numbers(snippet), value), (
        f"{path} quotes {snippet!r}, but the table now gives {' '.join(value)} "
        f"for {claim.name!r}")


def _documents() -> list[str]:
    """Every Markdown document and library module, outside generated and archived trees."""
    return sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                  if p.is_file() and p.suffix in (".md", ".py")
                  and not SKIPPED_DIRECTORIES & set(p.relative_to(ROOT).parts)
                  and (p.suffix == ".md" or p.relative_to(ROOT).parts[0] == "src"))


@pytest.mark.parametrize("retired", sorted(RETIRED))
def test_retired_value_does_not_reappear(retired: str) -> None:
    """A superseded number stays only in the documents that keep history."""
    pattern = re.compile(r"(?<![\w.])" + re.escape(retired))
    found = [path for path in _documents()
             if path not in RETIRED[retired] and pattern.search(_flat(_read(path)))]
    assert not found, f"superseded {retired!r} still quoted in {found}"
