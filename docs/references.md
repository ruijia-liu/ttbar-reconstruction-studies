# References for the conditional sliced-Wasserstein correction

The experiment label **COT** denotes a **grouped conditional sliced-Wasserstein loss**. The implementation partitions events using the frozen regression model's predicted mass, with boundaries fitted to the 60th and 90th training quantiles. Within each group, the four standardised targets are projected onto 32 fixed random unit directions; the predicted and true scalar projections are sorted independently and their mean squared differences are averaged over directions. The group contributions use the full-training group probabilities.

The closest algorithmic references are Deshpande et al. for learning with sliced Wasserstein objectives and Rowland et al. for projection-based estimation. Tabak et al. supplies the broader conditional-transport formulation. The three-group construction, frozen baseline, four-observable correction head and chosen hyperparameters are the implementation used in this project, rather than settings taken from any one of these papers.

## Recommended reading

1. **Ishan Deshpande, Ziyu Zhang and Alexander G. Schwing (2018). _Generative Modeling Using the Sliced Wasserstein Distance_. CVPR, pp. 3483–3491.**
   [Publisher PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Deshpande_Generative_Modeling_Using_CVPR_2018_paper.pdf) · [arXiv](https://arxiv.org/abs/1803.11188)

   The most direct starting point for the loss: high-dimensional distributions are compared through random one-dimensional projections, with empirical transport computed by sorted values. The paper studies generative modelling; this repository applies the distribution penalty to supervised four-observable regression.

2. **Mark Rowland, Jiri Hron, Yunhao Tang, Krzysztof Choromanski, Tamas Sarlos and Adrian Weller (2019). _Orthogonal Estimation of Wasserstein Distances_. AISTATS, PMLR 89, pp. 186–195.**
   [Proceedings and BibTeX](https://proceedings.mlr.press/v89/rowland19a.html) · [PDF](https://proceedings.mlr.press/v89/rowland19a/rowland19a.pdf)

   Explains efficient sliced-distance estimation using one-dimensional sorting and examines projection sampling. Our implementation uses 32 normalised Gaussian directions fixed by a seed. It does not implement the paper's orthogonal sampling scheme or its distinct projected-Wasserstein construction.

3. **Esteban G. Tabak, Giulio Trigila and Wenjun Zhao (2021). _Data driven conditional optimal transport_. Machine Learning 110, pp. 3135–3155.**
   [Journal article](https://doi.org/10.1007/s10994-021-06060-0) · [Open manuscript](https://arxiv.org/abs/1910.11422)

   Defines transport between source and target distributions conditioned on shared covariates. Here the condition is a discrete predicted-mass group. Our finite-projection penalty encourages agreement within those groups; it does not solve for the exact conditional Monge map described in the paper. The fitted correction network also depends on the paired regression objective.

4. **Gabriel Peyré and Marco Cuturi (2019). _Computational Optimal Transport_. Foundations and Trends in Machine Learning 11(5–6), pp. 355–607.**
   [Open monograph](https://arxiv.org/abs/1803.00567)

   Mathematical background for transport couplings, Wasserstein distances and numerical methods. Useful for understanding why equal-sized one-dimensional empirical samples can be matched by rank.

## What to cite for this implementation

For a short description, cite **Deshpande et al. (2018)** for the sliced objective and **Tabak et al. (2021)** for conditional transport. Add **Rowland et al. (2019)** when discussing finite random projections and sampling choices. Cite **Peyré and Cuturi (2019)** for general transport definitions.

A precise method description is:

> We regularise a frozen-feature regression correction using a grouped sliced-Wasserstein-2 squared loss. Events are assigned to fixed training-quantile groups of the baseline predicted mass; predicted and true four-observable vectors are compared through sorted one-dimensional projections within each group.

The paired regression term retains event correspondence. Sorting is used only for the distribution penalty and does not change stored event identities. The reported evaluation metric W1 is computed separately for each physical observable and differs from the standardised, four-dimensional, squared sliced penalty used during training.

BibTeX entries are available in [`references.bib`](../references.bib). Bibliographic metadata was checked against the linked primary sources on 16 November 2026.
