# Mathematical Foundations

This note contains the minimum theory needed to understand and defend the
project. The central problem is to train an energy-based model while using SVGD
to approximate the model samples required during training.

## 1. Energy-based model

An energy function assigns one scalar to every input:

\[
E_\theta(x): \mathbb{R}^d \rightarrow \mathbb{R}.
\]

It defines the probability density

\[
p_\theta(x)=\frac{\exp(-E_\theta(x))}{Z_\theta},
\qquad
Z_\theta=\int \exp(-E_\theta(x))\,dx.
\]

Low energy means high probability. The partition function \(Z_\theta\) is
usually intractable, which makes direct maximum-likelihood training difficult.

## 2. Score of an EBM

The score is the gradient of the log-density with respect to the input:

\[
s_\theta(x)=\nabla_x\log p_\theta(x).
\]

For an EBM,

\[
s_\theta(x)=-\nabla_x E_\theta(x),
\]

because \(Z_\theta\) depends on the model parameters but not on \(x\). This is
the key identity that allows gradient-based samplers to use an unnormalised EBM.

## 3. Langevin dynamics

Langevin dynamics combines movement towards high-density regions with random
exploration:

\[
x_i^{t+1}=x_i^t+\frac{\epsilon}{2}s_\theta(x_i^t)
              +\sqrt{\epsilon}\,\xi_i^t,
\qquad \xi_i^t\sim\mathcal{N}(0,I).
\]

The particles evolve independently. Noise helps them explore the distribution,
but finite chains may mix slowly or fail to move between distant modes.

## 4. Stein Variational Gradient Descent

SVGD updates a set of interacting particles:

\[
x_i^{t+1}=x_i^t+\epsilon\,\phi^*(x_i^t),
\]

where, for an RBF kernel \(k\),

\[
\phi^*(x_i)=\frac{1}{n}\sum_{j=1}^{n}
\left[k(x_j,x_i)s_\theta(x_j)+\nabla_{x_j}k(x_j,x_i)\right].
\]

The first term attracts particles towards high-probability regions. The second
term repels nearby particles and prevents them from collapsing to the same
point. Unlike Langevin, vanilla SVGD is deterministic after initialisation and
its kernel calculation costs \(O(n^2)\) for \(n\) particles.

## 5. Contrastive training of the EBM

The maximum-likelihood gradient can be written as the difference between a data
expectation and a model expectation. A practical energy loss is

\[
\mathcal{L}(\theta)=
\mathbb{E}_{x^+\sim p_{\mathrm{data}}}[E_\theta(x^+)]-
\mathbb{E}_{x^-\sim p_\theta}[E_\theta(x^-)],
\]

where \(x^+\) are observed data and \(x^-\) are negative samples from the
current model. Minimising this expression lowers the energy of real data and
raises the energy of model-generated negatives.

Exact negative samples are unavailable. The project therefore compares two
approximations of \(x^-\): Langevin dynamics and SVGD. The model architecture,
optimiser, data and experimental budget must otherwise remain fixed.

## Project hypothesis

The repulsive interaction in SVGD may provide more diverse negative samples and
better mode coverage than independent finite-step Langevin chains. This benefit
may be strongest with few particles, while SVGD may be slower and sensitive to
the kernel bandwidth.

## Day 1 self-check

Before implementing the main experiment, be able to answer these questions in
your own words:

1. Why does low energy correspond to high probability?
2. Why does the partition function disappear from the score?
3. What is the difference between differentiating with respect to \(x\) and
   differentiating with respect to \(\theta\)?
4. What are the two terms in the SVGD direction, and what does each one do?
5. Why are negative samples necessary when training an EBM?
6. Why is Langevin included if the main method of the project is SVGD?
7. What observation would support the project hypothesis, and what observation
   would contradict it?

## Primary references

- Qiang Liu and Dilin Wang, [Stein Variational Gradient Descent](https://arxiv.org/abs/1608.04471).
- Yang Song and Diederik P. Kingma, [How to Train Your Energy-Based Models](https://arxiv.org/abs/2101.03288).
- Priyank Jaini, Lars Holdijk, and Max Welling, [Learning Equivariant Energy Based Models with Equivariant SVGD](https://arxiv.org/abs/2106.07832).
