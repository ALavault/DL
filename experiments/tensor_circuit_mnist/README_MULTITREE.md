# Exact multi-tree tensor circuit experiment

This cycle tests whether the main limitation of the single Morton tree is its
factorization topology rather than its raw parameter count.

Nine exact rank-16 circuits are trained:

- six spatial decompositions with the same seed;
- four replicas of the centered Morton decomposition with different seeds
  (the first replica belongs to both groups).

The final stage compares, at nearly the same parameter budget as the previous
single rank-32 circuit:

1. a four-component mixture of heterogeneous spatial trees;
2. a four-component mixture of the same tree trained with different seeds;
3. the best four-component subset among all candidates;
4. joint maximum-likelihood fine-tuning of the heterogeneous and homogeneous
   mixtures.

All component densities and their mixtures are normalized by construction.
The subset and mixture weights are selected only on the validation split; the
test split is read after selection.
