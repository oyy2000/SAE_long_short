# Sparse Autoencoders: A Larger Dictionary, Sparse Activation

**Input → sparse features → reconstruction**

- At each token position, the SAE encodes a **3,584-dimensional hidden state** into **28,672 feature activations**: an **8× expansion**.
- With **TopK = 64**, at most **64 features have positive activation** at that position. All remaining features are zero.
- The larger dictionary offers more candidate directions; sparsity limits how many directions represent each token.
- The decoder reconstructs the hidden state as a weighted sum of feature directions, plus a bias.

**Illustrative example — five candidate features, k = 2**

```text
Encoder scores:       [3.0, -0.5, 1.2, 4.0, 0.2]
Keep Top-2, then ReLU: [3.0,  0.0, 0.0, 4.0, 0.0]
Reconstruction:       b_dec + 3.0 × d_0 + 4.0 × d_3
```

**Explained variance** measures reconstruction quality: `1 − reconstruction squared error / mean-baseline squared error`. An EV of 0.774 means a 77.4% reduction in that error relative to predicting the mean; it is not answer accuracy or a measure of semantic correctness.

Feature IDs identify learned directions, not vocabulary tokens. “Short-associated” and “long-associated” describe activation differences; causal effects require intervention experiments.
