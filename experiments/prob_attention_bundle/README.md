# Probabilistic attention MNIST source bundle

The four `part-*` files concatenate to a base64-encoded `tar.gz` archive. The workflow reconstructs `experiments/prob_attention_mnist/` before running.

The experiment compares four exact autoregressive densities on a bijective 2x2-patch representation of static binarized MNIST:

- `fixed-prob`: position-only probability-state routing;
- `bayes-prob`: content-dependent latent-address attention with arithmetic pooling in the probability simplex;
- `bayes-logit`: the same address posterior with geometric/logit pooling;
- `transformer`: a conventional causal Transformer of comparable size.

The code reports exact NLL, throughput, classifier-feature quality, precision/recall, binary topology, and sample grids. The local source archive used to create these parts is retained with the experiment artifacts.