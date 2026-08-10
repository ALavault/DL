# Static binarized MNIST tensor-circuit scale test

This experiment evaluates an exact normalized hierarchical probability circuit
on the standard static binarized MNIST split (50k train / 10k validation / 10k
test). The 28x28 variables are centered in a 32x32 Morton tree; the 240 padded
leaves are marginalized analytically.

The workflow trains four circuit variants in parallel. Deep baselines are not
reimplemented: the final analysis compares the exact NLL with values reported
in their original papers.
