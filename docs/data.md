# Data, checkpoints and full-run reproduction

This repository contains code, configurations, aggregate results and figures.
It does not bundle event datasets, trained checkpoints, cached representations
or per-event prediction files. The small Barron partition-function lookup table
is an implementation asset, not a dataset.

For the recorded COT experiment, the inputs were 4,692,288 training events and
260,190 validation events. The file identifiers below are SHA-256 digests of
the exact input artifacts used in that experiment:

| Artifact | SHA-256 |
|---|---|
| Training event file | `5317a849d25b0b721b33ab9e8b0ee5bf44709238b589ebeeb6ac21b0262e9bd2` |
| Validation event file | `da630904b6c91a8afefe5a4fa96a0fd8a9eaac537b3e7f975efdca31eecc28ff` |
| Canonical four-observable array | `8cab939641f4a45fd79e5fa107229ee8b31f51097a573e42967ae55a1b0e83b8` |

To rerun a historical sweep, preserve the archive directory names, install the
upstream dependencies in its documented environment, and supply the expected
data and model artifacts at the paths described by its runner. Local machine
and cluster account paths have been removed. `experiments/provenance.json`
records source hashes and the transformations made to the distributed code.

For COT, cache each event's frozen 128-dimensional encoder feature and four
standardised baseline outputs, with the paired four truth values in identical
row order. Fit standardisation and mass-group boundaries on training events
only. Preserve the original physical-unit baseline predictions for export as
`baseline_physical + correction_z * training_std`. Reusing the same event count
does not establish alignment: verify event identifiers or the canonical input
hash and exact row order before joining artifacts.

Large artifacts can be shared separately as versioned archives. Each shared
archive should include a relative-path manifest with size, SHA-256, training
configuration and parent baseline identifier. No external artifact download is
required for the portable COT example or unit tests.
