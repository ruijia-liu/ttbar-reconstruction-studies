# Conditional projected distribution matching

The COT experiment refines a frozen Huber + 0.1 EMD regression model for four
dileptonic top-pair observables: pair mass, rapidity difference, positive-lepton
helicity cosine and dilepton opening-angle cosine. A small trainable head
receives the frozen event representation and four baseline predictions. Its
bounded correction is added in the original training-standardised coordinates.
The head is supervised by the paired truth and a distribution penalty.

## Fixed groups

For each event, use the **frozen baseline predicted mass**. Fit the 60th and 90th
percentiles on the complete training split and persist them with the training
group counts and probabilities. The resulting populations are approximately
60%, 30% and 10%; ties can change these fractions. Equality at a boundary enters
the upper group. Boundaries and assignments do not follow the evolving corrected
prediction. Validation/test data are never used to fit the boundaries.

These groups are distinct from the physical bins used to plot migration matrices.
For each batch group, one event-index mask selects both corrected predictions
and their corresponding truth values. The truths are **not independently grouped
by true mass**. This is a coarse conditioning variable derived from detector-side
information, rather than conditioning the discrepancy on every measured input.

## Project, sort and compare

Write the four standardised prediction and truth as $\hat{\mathbf z}_i$ and
$\mathbf z_i$. With 32 fixed unit directions $\mathbf v_\ell$, form the scalar
projections $\mathbf v_\ell^T\hat{\mathbf z}_i$ and $\mathbf v_\ell^T\mathbf z_i$.
Within batch group $g$, sort each projected list independently and denote the
ordered entries by $a_{g\ell,(j)}$ and $b_{g\ell,(j)}$. The implemented penalty is

$$
\mathcal L_{\mathrm{COT}}=
\sum_{g:n_g\ge2}\frac{\pi_g}{32n_g}
\sum_{\ell=1}^{32}\sum_{j=1}^{n_g}
\left(a_{g\ell,(j)}-b_{g\ell,(j)}\right)^2,
$$

where $n_g$ is the batch group size and $\pi_g$ its full-training population
fraction. Empty and singleton groups contribute zero, without renormalising the
other weights. The batch objective is not asserted to be an unbiased estimate of
the population transport distance. Standardisation is fitted on training data
and keeps the mass coordinate from dominating merely because it is measured in
GeV. The reference implementation uses float32 loss arithmetic, including under
mixed-precision training, and fixed projection seed 4201.

Sorting pairs ranks instead of event identities. The stored data order is
unchanged. Gradients flow back to each contributing prediction; the target is
detached. A permutation of predictions within a group can leave this loss at zero
while producing large event errors. The paired regression term is therefore
retained. In the correction experiment the total objective is

$$
\mathcal L=\mathcal L_{\mathrm{base}}+0.1\mathcal L_{\mathrm{COT}},
$$

where $\mathcal L_{\mathrm{base}}=2\mathcal L_{\mathrm{Huber},\delta=1}+0.1\mathcal L_{\mathrm{marginal}\,W_1}$,
with mean reduction over batch events and four coordinates. The factor of two
preserves the recorded correction experiment.
This finite-projection grouped loss does not learn an optimal Monge map and is
not the univariate $W_1$ used for evaluation. Thirty-two directions supply a
tractable joint regulariser, not complete identification of a four-dimensional
distribution.

## Using the module

```python
import json
from ttbar_reco import fit_groups, assign_groups, make_directions, conditional_ot_loss

reference = fit_groups(full_training_baseline_mass)
with open("groups.json", "w") as stream:
    json.dump(reference.to_dict(), stream, indent=2)

groups = assign_groups(batch_baseline_mass, reference)
directions = make_directions().to(prediction_z.device)
cot = conditional_ot_loss(prediction_z, target_z, groups,
                          reference.probabilities, directions)
loss = base_loss + 0.1 * cot
loss.backward()
```

Save `reference.to_dict()` and the direction tensor with the model checkpoint.
Restore group metadata using `GroupDefinition.from_dict(...)`; do not refit it
on validation or test events. The Python API validates metadata, but the caller
remains responsible for passing only training rows to `fit_groups`.

Run the synthetic example with `python examples/cot_minimal.py` after installing
the package. It demonstrates the paired Huber, marginal W1 and COT terms with
the recorded objective weights; its simple correction head is illustrative and
does not replace the full recorded training pipeline.

## Evaluation

Evaluate all metrics on one unchanged output per event: residual quantile width,
truth-row-normalised migration matrices, marginal $W_1$, bias and physical-support
violations. Keep the original baseline and a correction-only head as separate
controls. Report low/high-mass slices and the same bin boundaries for every
method. The recorded study used validation-selected checkpoints and one seed.

## Literature

The projected, sorted-sample construction follows the sliced-Wasserstein
literature. The fixed predicted-mass grouping is this project's application of
that construction, rather than a reproduction of a named conditional-transport
model from the cited papers.

1. I. Deshpande, Z. Zhang and A. G. Schwing, *Generative Modeling Using the Sliced
   Wasserstein Distance*, CVPR (2018), 3483–3491.
   [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Deshpande_Generative_Modeling_Using_CVPR_2018_paper.pdf)
2. M. Rowland et al., *Orthogonal Estimation of Wasserstein Distances*, AISTATS
   (2019), PMLR 89, 186–195.
   [Paper and metadata](https://proceedings.mlr.press/v89/rowland19a.html)

The present implementation uses fixed Gaussian unit directions, not the
orthogonal estimator investigated in reference 2. Both sources provide relevant
context; no claim of implementing every construction from either paper is made.
