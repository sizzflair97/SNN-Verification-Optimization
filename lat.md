# LIF-TTFS Extension: Model Semantics

## Scope

The LIF extension keeps the finite-horizon TTFS classifier used by the
existing IF model. Only the neuronal dynamics are replaced by the
current-based LIF dynamics of Göltz et al.; the output decoder and its
tie-breaking rule remain unchanged.

## Finite-horizon spike times

Let $T$ be the inference deadline. For every neuron, an analytical LIF
threshold crossing is an actual spike only when it occurs before the
deadline:

$$
s_i =
\begin{cases}
t_i, & \text{if the neuron crosses the threshold at } t_i < T,\\
T,   & \text{otherwise.}
\end{cases}
$$

For hidden neurons, the value $T$ is only a finite no-spike sentinel:
the neuron remains marked as silent and emits no synaptic event to the next
layer. At the output layer, the decoder interprets $T$ as the same
terminal pseudo-spike used by the IF classifier.

## Classification and ties

The prediction is computed only from the finite output times:

$$
\hat y = \operatorname*{arg\,min}_{k \in \{0,\ldots,K-1\}} s_k^{(L)}.
$$

The implementation uses the standard lowest-index argmin rule. Therefore:

- a tie between actual output spikes is won by the lowest class index;
- a tie between actual and terminal spikes at $T$ is resolved the same way;
- if every output neuron is silent, all output times equal $T$, so class
  $0$ is selected.

This class-0 outcome is an explicit part of the classifier definition, not
a membrane-potential fallback. It is applied identically to the IF and LIF
models so that their verification problems remain directly comparable.

For a predicted class $y$, an index-aware sufficient robustness condition
is

$$
u_y < \ell_k \quad (k < y), \qquad
u_y \le \ell_k \quad (k > y),
$$

where $[\ell_k,u_k]$ bounds output time $s_k^{(L)}$. The strict inequality
against lower-index classes accounts for their advantage under a tie.

## Relation to Göltz et al.

The extension adopts the Göltz current-based LIF dynamics and analytical
first-threshold-crossing calculation. The finite deadline, terminal
pseudo-spike, and lowest-index decoder are conventions of this work. Thus
the model should be described as a Göltz-style LIF neuron embedded in our
finite-horizon TTFS classifier, rather than an unchanged reproduction of
the original Fast&Deep readout.
