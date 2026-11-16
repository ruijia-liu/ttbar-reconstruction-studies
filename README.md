# Dileptonic top-pair reconstruction studies

This repository brings together the full sequence of my dileptonic top-pair
reconstruction studies: data preparation, VyPER reproduction, direct-regression
benchmarks, architecture and hyperparameter searches, alternative losses,
mass-dependent weighting, probabilistic models, physics-informed tasks and
distribution corrections. It contains the code, configurations and evaluation
procedures used to explore these approaches, including ideas that did not
improve the overall result.

The four target observables are $m_{t\bar t}$, $\Delta|y|$,
$\cos\theta_K^+$ and $\cos\varphi$. The work builds on
[HepSetReg](https://github.com/els285/HepSetReg/tree/bsc) and
[VyPER](https://github.com/tzuhanchang/VyPER).

## Research questions

The comparisons ask how changes to the model and training objective affect:

- **Event resolution:** how narrowly the prediction errors are distributed.
- **Migration:** how often events remain in the correct physics bins.
- **Distribution agreement:** how closely the predicted spectra follow truth.

MAE, RMSE, bias, high-mass behaviour and physical-range violations provide
additional diagnostics. The same predictions must be used for all these checks;
improving a distribution alone does not establish better event reconstruction.

## Research progression

| Stage | Approaches explored | Code |
|---|---|---|
| Data and regression benchmarks | Four-observable preparation, row alignment, MSE and MSE + EMD | [`initial_regression`](experiments/initial_regression) |
| VyPER reproduction | Full training, validation prediction and reconstruction of the four observables | [`vyper_reproduction`](experiments/vyper_reproduction) |
| Architecture and parameters | Transformer and ParT; learning rate, width, depth, heads, batch size, regularisation, scheduling and EMD weight | [`transformer_sweep`](experiments/transformer_sweep_20261109) |
| Loss functions | MSE, Huber, log-cosh, marginal EMD and sliced-Wasserstein combinations | [`loss_sweep`](experiments/loss_sweep_20261110) |
| Mass-dependent weighting | Learned high-mass probability, fixed mixtures and truth-region controls | [`agent_sweep`](experiments/agent_sweep_20261111) |
| Probabilistic predictions | Gaussian and RealNVP heads; single samples, averages and deterministic predictions | [`probabilistic_sweep`](experiments/probabilistic_sweep_20261111_retry2) |
| Prediction mixing | Fixed mixtures of deterministic predictions and flow samples | [`mixture_scan`](experiments/mixture_scan_20261112) |
| Targeted objectives | Soft migration-bin loss and adaptive robust loss | [`objective_sweep`](experiments/objective_sweep_20261112) |
| Model structure and physics | Fixed/learned two-expert mixtures and auxiliary top/antitop reconstruction | [`structure_sweep`](experiments/structure_sweep_20261112) |
| Distribution corrections | Correction-only control, conditional OT, metric constraints, OT + constraints, RankSim and FlowAlign | [`methods_sweep`](experiments/methods_sweep_20261112) |
| Common evaluation | Resolution, migration matrices, distribution errors and audit of saved predictions | [`paper_target`](experiments/paper_target_20261111) |

The [experiment index](docs/experiments.md) gives the number of comparisons,
parameters, objectives, dependencies and the connections between campaigns.
Conditional OT is one of the approaches in the final correction study. Its
reusable module and separate HepSetReg contribution are linked below.

## Start here

| Contents | Location |
|---|---|
| All experimental methods and how the runs connect | [`docs/experiments.md`](docs/experiments.md) |
| Historical training, evaluation and plotting scripts | [`experiments`](experiments) |
| Parameters for the recorded sweeps | [`configs`](configs) |
| Current six-method correction results and figures | [`results`](results), [`figures`](figures) |
| Data and checkpoint requirements | [`docs/data.md`](docs/data.md) |
| Implementation checks | [`docs/validation.md`](docs/validation.md) |
| Reusable COT module and explanation | [`src/ttbar_reco`](src/ttbar_reco), [`docs/conditional_ot.md`](docs/conditional_ot.md) |
| COT literature and BibTeX | [`docs/references.md`](docs/references.md), [`references.bib`](references.bib) |

## Running the code

Start with the campaign you want to study in the [experiment index](docs/experiments.md).
Each archived campaign keeps its matching model, loss, training, evaluation and
plotting scripts together. Full-data runs require the external event files,
training statistics and, where applicable, a saved baseline and feature cache.
The index explains these dependencies and provides a runner example.

The separately packaged COT module also has a short CPU example:

```bash
python -m pip install -e ".[test]"
python examples/cot_minimal.py
python -m pytest
```

This example uses synthetic events to demonstrate the loss. It does not rerun
the historical training campaigns. The [HepSetReg contribution](https://github.com/els285/HepSetReg/pull/6)
contains an example that connects the correction to a frozen regression model.

## Results and interpretation

The archive retains both useful changes and unsuccessful comparisons so that
the progression can be inspected. Checkpoint-selection rules evolved during
the study and are recorded in each campaign's plan; they need to be matched
when attributing an improvement to a particular model or loss.

The aggregate results currently bundled in [`results`](results) cover the last
six-method correction comparison, with distributions and migration matrices in
[`figures`](figures). Earlier campaigns retain their analysis and plotting code
in their respective directories; their complete prediction artifacts are not
bundled. The recorded comparisons use validation data and do not establish
superiority to a paper under matched test conditions.

## Code and references

Method-specific source notes are kept beside the corresponding implementations,
including adaptive robust loss, RankSim and FlowAlign. The
[COT reference note](docs/references.md) explains the projected transport
construction and distinguishes it from the project's grouping and correction
design. Citations and implementation details should be read for the particular
method being used.

Upstream packages retain their own authorship and licences. The adapted RankSim
file retains its CC BY-NC-SA 4.0 notice. See [`LICENSE`](LICENSE) for the scope of
the MIT licence on the remaining original project code.
