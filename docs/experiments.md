# Experiment code index

The `experiments/` directory preserves the training, model, loss, evaluation and plotting code used during the study. Campaign dates identify the original experiments. The reusable conditional distribution-matching implementation is documented separately in [conditional_ot.md](conditional_ot.md).

These historical runners expect the original data schema, upstream packages and saved artifacts. They are an archive of the experimental procedures, rather than a single command that downloads data and reproduces every run. Numerical settings are recorded in `configs/`; the fuller `experiment_plan.json` beside each runner also retains the original artifact hashes and evaluation settings.

## Included campaigns

| Directory under `experiments/` | What was tried | Main entry points |
|---|---|---|
| `initial_regression` | Four-observable data conversion, MSE and MSE + EMD training, validation inference and paired-event diagnostics | `prepare_ttbar_four.py`, `fix_ttbar_dtype.py`, `train_ttbar_four.py`, `predict_ttbar_validation.py`, `compare_ttbar_validation.py`, `ttbar_metrics.py` |
| `vyper_reproduction` | Official VyPER training wrappers, bounded-memory graph-cache concatenation, explicit resume, validation prediction, jet–lepton pairing and reconstruction of the four observables | `prepare_vyper_full.py`, `train_vyper_full_100.py`, `predict_vyper_validation.py`, `evaluate_vyper_validation.py`; physics helper under `benchmark_20261027/part_slot_audit/` |
| `transformer_sweep_20261109` | 26 planned comparisons: ordinary Transformer and ParT, each with baseline and 12 single-factor variants | `run_trial.py`, `compare.py`, `summarize_round.py`; two plotting scripts |
| `loss_sweep_20261110` | Seven objectives: MSE, MSE + 0.1 EMD, Huber thresholds 0.5 and 1, log-cosh, Huber + 0.1 EMD, Huber + 0.1 sliced Wasserstein | `losses.py`, `module.py`, `run_trial.py`, `analyze_results.py` |
| `agent_sweep_20261111` | Five mass-dependent loss controls: Huber, fixed 0.5 mix, fixed mean mix, truth-region weighting, learned predicted high-mass probability | `gate_model.py`, `train_gate.py`, `gated_loss.py`, `run_trial.py` |
| `probabilistic_sweep_20261111_retry2` | Gaussian and conditional RealNVP density heads on a frozen regression encoder; deterministic, sampled and averaged-prediction evaluation | `cache_features.py`, `density_heads.py`, `run_density.py`, `evaluate_density.py` |
| `mixture_scan_20261112` | Fixed mixtures of an existing deterministic prediction and one flow sample, with coefficients 0, 0.1, 0.2, 0.35 and 0.5 | `analyze_mixture.py` |
| `objective_sweep_20261112` | Huber + EMD control, additional soft migration-bin loss, and a bounded adaptive robust objective + EMD | `migration_loss.py`, `adaptive_loss.py`, `module.py`, `run_trial.py` |
| `structure_sweep_20261112` | Control, fixed two-expert mixture, learned two-expert mixture, and auxiliary top/antitop four-vector reconstruction | `models.py`, `physics_aux.py`, `aux_dataset.py`, `run_trial.py` |
| `methods_sweep_20261112` | Six deterministic correction-head methods: control, conditional OT, metric constraints, OT + constraints, RankSim and FlowAlign | `common.py`, `ot_proxy.py`, `ranksim.py`, `flowalign.py`, `prepare_data.py`, `train_field.py`, `run_trial.py`; analysis and plot scripts |
| `paper_target_20261111` | Common audit of saved predictions and the explicit local comparison protocol | `audit_existing.py`, `evaluation_protocol.json` |

The last successful probabilistic runner revision is retained as `retry2`; earlier deployment retries are not separate scientific methods. Preparation programs that generated or copied deployment scripts are omitted, while their resulting training code and configuration are retained. Login, file-transfer and submission-control utilities are not part of this archive.

## Parameters and method details

The Transformer sweep varies learning rate (`1e-4`, `3e-4`, `1e-3`), hidden dimension (64 or 128), layers (3 or 6), heads (4 or 8), batch size (1024, 2048 or 4096), dropout (0 or 0.1), weight decay (0 or 0.01), constant versus warmup/cosine scheduling, and EMD weight (0, 0.05 or 0.1). This is a baseline plus single-factor variants within each architecture, not a full Cartesian grid.

The loss sweep explicitly controls scaling: MSE is the mean squared residual, Huber is multiplied by 2, and log-cosh by twice the squared scale. Marginal EMD sorts predictions and targets separately for each standardized observable. The joint sliced loss projects the four-dimensional standardized outputs onto fixed unit directions and compares their sorted projections. The exact squared/absolute convention is specific to each loss: the later conditional OT objective uses squared projection differences, while the earlier sliced-Wasserstein loss uses absolute differences.

The high-mass gate is trained with reconstructed input features and a truth-mass label above 800 GeV. Two-fold out-of-fold probabilities set the training weights. Truth is used for supervision, not as an inference input. The separate truth-region arm is a control for loss weighting.

The probabilistic study fits conditional densities in transformed coordinates: log mass, unchanged rapidity difference and atanh of the two cosines. It records train-only transformation statistics and validates the frozen encoder cache. Density fitting and inference choices are separate: a single sample, an average of samples and a deterministic location can have different resolution and distribution behavior.

The migration loss assigns differentiable soft-bin probabilities using logistic boundaries, including underflow and overflow categories. Training-bin frequencies determine its row weights. The adaptive robust arm learns four shape parameters constrained to `(0.1, 1.9)` at fixed standardized scale, with the alpha-dependent log partition function included. `barron_partition.npz` is a small reproducible quadrature table, not an event dataset; `test_adaptive_loss.py --build-table` regenerates it.

The expert models share an encoder and use two prediction heads. Both fixed and learned routing controls have the same specialist supervision; the router is the changing component. The physics auxiliary predicts top and antitop momenta and positive masses, then imposes consistency on mass and rapidity. It does not impose angular consistency.

The six-method comparison uses the same frozen encoder and a trainable correction head. Its 132 inputs combine a 128-dimensional embedding with the four baseline standardized predictions. Conditional OT groups events by fixed training quantiles of baseline predicted mass (bottom 60%, next 30%, top 10%) and matches joint projected distributions within those groups. The physics migration bins serve evaluation and the migration objective; they are not the conditional OT groups. See [conditional_ot.md](conditional_ot.md) for the reusable formulation and [references.md](references.md) for the papers.

## Data and external dependencies

The historical train/validation studies use 4,692,288 training events and 260,190 validation events, with the ordered targets:

```text
ttbar_mass, delta_abs_y, cos_K_plus, cos_phi
```

ROOT/HDF5/Parquet event files, model checkpoints, prediction arrays, feature caches and result directories are not bundled. They must be supplied separately, with permission to use and redistribute them. Small source configuration files and the numerical partition table are included. The original data-conversion and row-alignment checks remain in the archive.

Install [HepSetReg's `bsc` branch](https://github.com/els285/HepSetReg/tree/bsc) separately. The archived runners import its models, dataset and training interfaces; core upstream files are not copied here. Some runs record exact source hashes, so a later upstream checkout may fail an identity check even when its public API looks similar. For a new experiment, record the chosen upstream revision, create a new manifest and regenerate its hashes against the actual inputs. Do not relabel the historical hashes as verification of different artifacts.

VyPER reproduction additionally requires the [official VyPER implementation](https://github.com/tzuhanchang/VyPER) and its dependencies. The wrappers call that installation and preserve the original run layout. Paths formerly under a home scratch directory now resolve below `RECO_DATA_ROOT` (default `external`). The local evaluator and plotting scripts expect the original `hepsetreg_tools`/reference-artifact layout, so map those paths to your restored artifacts before executing them.

The archive uses Python packages including PyTorch, NumPy, SciPy, Awkward Array, PyArrow, PyTorch Lightning, Matplotlib, h5py, PyYAML, Uproot and Vector. VyPER's runtime supplies its Hydra/OmegaConf configuration dependencies. These historical dependencies are broader than those needed by the reusable COT module.

## Running an archived campaign

1. Inspect the campaign's runner and `experiment_plan.json`, and restore the exact input schema and required artifacts. Change `external/...` placeholders to your own data locations in a new run configuration.
2. Supply a separately installed HepSetReg checkout for runners with `--repo`, and the prepared train/validation Parquet directory for `--data`. Run from the campaign directory so its local imports resolve to the matching version.
3. For density heads, first generate and verify the frozen cache with `cache_features.py`. For the final six methods, first run `prepare_data.py`; FlowAlign also needs the fitted field produced by `train_field.py`.
4. Execute GPU training in a Slurm allocation. Several runners explicitly require `SLURM_JOB_ID` and an available CUDA GPU. Use the cluster's own partition/account configuration and a 30-minute job limit; the original runners reserve part of that budget for full validation and export.
5. Keep output directories separate from the source and inspect the status files, completed epochs, event counts and identity checks before comparing metrics.

An example invocation for an end-to-end campaign, after updating its plan and supplying the required canonical validation artifact, is:

```bash
cd experiments/loss_sweep_20261110
python run_trial.py --index 0 --repo "$HEPSETREG_REPO" \
  --data "$RECO_DATA_DIR" --output "$RECO_OUTPUT_DIR"
```

This is a runner example, not a complete submission script. It does not request a GPU or stage external data.

## Evaluation and tests

`evaluation.py` in the later campaigns computes all primary metrics from the same continuous event predictions. Resolution is the full central residual width `Q(0.8413) - Q(0.1587)`, not its half-width. Migration uses fixed truth rows, includes predictions outside the displayed range in each row denominator, and reports the unweighted mean diagonal. The distribution metric is one-dimensional Wasserstein distance for each observable. MAE, RMSE, bias, mass-tail checks and physical-range violations are additional diagnostics.

Checkpoint selection changed during the study. Early sweeps select minimum full-validation standardized MSE. Later campaigns first seek candidates with all four resolution and all four migration measures no worse than the recorded reference, then select distribution agreement; their code defines the fallback when none qualifies. The later runners also retain MSE-selected checkpoints. Compare matching selection rules when isolating the effect of an objective.

The `test_*.py` files are included beside each implementation. Run tests separately within each campaign directory because names such as `losses.py` and `evaluation.py` intentionally preserve their historical versions. Tests cover loss values and gradients, cache integrity, prediction export, migration edge cases, adaptive-loss normalization, routing and physical reconstruction. A useful per-campaign command is:

```bash
python -m unittest discover -s . -p 'test_*.py'
```

These checks do not stand in for a full-data GPU rerun. The archive was prepared without retraining its campaigns; artifact-dependent integration checks require the omitted external files. The original validation-selection design is preserved in the plans, not converted into an independent test evaluation.

## Provenance and licenses

[`experiments/provenance.json`](../experiments/provenance.json) records each original relative path, original SHA-256, published SHA-256 and transformation note. Publication changes remove workstation-specific import paths, replace private filesystem locations with configurable or relative placeholders, and extract parameter-only configuration views. Existing source attributions are retained.

`methods_sweep_20261112/ranksim.py` adapts Royal Bank of Canada's RankSim implementation and is licensed **CC BY-NC-SA 4.0**. Its header and `method_sources.txt` preserve copyright, source links and modifications. That file's noncommercial and share-alike terms are separate from the license of this project's own code. The official RankSim source credits Blackbox Differentiation of Combinatorial Solvers; that attribution is retained in the method notes.

The Barron and FlowAlign modules are independent equation-based implementations, as recorded in `adaptive_sources.txt` and `method_sources.txt`. Their official source code is not bundled. HepSetReg and VyPER remain external dependencies with their own distribution terms; adding wrappers here does not grant a license to those projects.
