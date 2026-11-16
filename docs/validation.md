# Publication checks

The following checks were run on the distributed historical archive on 16 November 2026. They concern the archived implementations; the portable COT package has a separate test suite under `tests/`.

## COT package and HepSetReg contribution

The portable package passed all 13 tests, including 100 random comparisons with
the archived objective. The HepSetReg contribution passed all 14 focused tests,
including the frozen-baseline correction integration. Both synthetic examples
ran successfully. An additional 200-case comparison of the two loss modules
against the original experiment gave identical forward values and prediction
gradients (maximum difference zero). Group metadata, event assignments and the
default projection directions also matched. These checks used Python 3.12,
PyTorch 2.4.1 CPU and NumPy 1.26.4.

## Static and provenance checks

- All 89 archived Python files parsed successfully, and all included JSON files parsed successfully.
- All 107 original/derived entries in `experiments/provenance.json` matched their published SHA-256 digests.
- The archive, extracted configurations and experiment index were checked for private workstation/account paths and credential strings. Those paths were removed or replaced by relative/configurable placeholders.
- No event datasets, model checkpoints or per-event prediction arrays were added. `barron_partition.npz` is a numerical quadrature asset.

## Archived unit tests

| Campaign | Observed result |
|---|---|
| Initial regression | All 36 tests passed. |
| Regression losses | All 9 tests passed. |
| High-mass gate | All 14 tests passed. |
| Six correction methods | 28 tests passed; one Parquet-export test was blocked by the available Awkward/PyArrow environment. |
| Migration/adaptive objectives | 27 tests passed; the adaptive-loss and evaluation test modules could not load with the available SciPy dependency combination. |
| Expert/physics structure | 36 tests passed; the evaluation test module and two Parquet-based physics tests were blocked by the available SciPy/Awkward/PyArrow dependency combination. |

The complete first three suites therefore account for 59 passing tests. The later rows report partial coverage explicitly and should not be read as complete suite passes. Mathematical checks that did execute used PyTorch 2.4.1 on CPU. The blocked checks required packages from a separate environment whose SciPy and PyArrow builds required NumPy 2, while the available PyTorch test environment used NumPy 1.26.4. These are unresolved environment checks, not demonstrated algorithm failures or successful tests.

The probabilistic cache-contract tests require a compatible external HepSetReg checkout; they were retained but not rerun for publication. No full-data GPU training, VyPER reproduction, checkpoint-dependent inference, or independent test-set evaluation was performed as part of these publication checks.

Run each archive's tests from its own directory, with a compatible dependency set, because modules with the same names intentionally preserve different campaign versions. See [experiments.md](experiments.md) for dependencies and execution requirements.
