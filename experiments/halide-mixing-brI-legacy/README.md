# halide-mixing-brI-legacy

## Purpose

Run the correction on Marc Meléndez's legacy ~20,000-walker Br-I halide-mixing
diffmaps -- a kinetic-Monte-Carlo bed of excitons on a halide-disorder landscape
-- as cross-material coverage and a bounded-fit-vs-moment comparison.

The correction is inactive on this short window, so the bed is coverage, not a
stress test. The clean high-statistics Br-I bed is
`experiments/halide-mixing-highn/`.

## Contents

- `br_data.py` -- loader; reads the diffmaps directly from the pshare.
- `br_halide_validation.py` -- Gaussian fit + `subtract` + floor cut; naive vs
  corrected vs particle truth across three Br fractions and a CNR0 sweep.
- `rerun/` -- instructions and helpers for running and averaging multiple
  simulation instances (`average_diffmaps.py`, `run_instances.sh`, `README.md`).

## Data

Pshare only; see `DATASETS.md` (`P:/Marc/shared_data/Halide_mixing`). No local
copy.

## Run

```bash
.venv/Scripts/python.exe experiments/halide-mixing-brI-legacy/br_halide_validation.py
```

Outputs to repo-root `output/`; committed snapshot is here.

## Key result + figure

Halide mixing suppresses transport (apparent alpha 0.97 -> 0.90, D 3-10x below
the clean-layer D0 as the Br fraction rises) and the bounded fit recovers the
carrier-truth alpha where the moment estimator reads it low (the 150 nm box clips
the moment's tails). The 0.1 ns window keeps every frame's per-frame CNR above
the floor, so the correction is inactive (naive ~ corrected). Figure:
`br_halide_validation.png`.

## Depends on

`width_debias`. Local sibling loader `br_data.py`.

## See also

`experiments/halide-mixing-highn/`.
