


### Note about the ACF
*Written by Claude, not checked yet.*

Let $`I(\vec{x})`$ be the intensity of the damage channel at pixel $`\vec{x}`$,
and let $`M`$ be the set of pixels inside the leaf mask, containing $`N=|M|`$
pixels, with indicator function $`m(\vec{x})=1`$ if $`\vec{x}\in M`$ and $`0`$
otherwise. The mean and variance of the damage signal *within the leaf* are

```math
\mu = \frac{1}{N}\sum_{\vec{x}\in M} I(\vec{x})
\qquad\qquad
\sigma^2 = \frac{1}{N}\sum_{\vec{x}\in M} \left(I(\vec{x})-\mu\right)^2
```

For a displacement vector $`\vec{X}`$, the number of pixel pairs separated by
$`\vec{X}`$ that have *both* pixels inside the leaf is

```math
n(\vec{X}) = \sum_{\vec{x}} m(\vec{x})\, m(\vec{x}+\vec{X})
```

and the ACF is the mean product of the two pixels' deviations from $`\mu`$,
normalized by the variance:

```math
\mathrm{ACF}(\vec{X}) = \frac{1}{\sigma^2\, n(\vec{X})}
    \sum_{\vec{x}\in M,\ \vec{x}+\vec{X}\in M}
    \left(I(\vec{x})-\mu\right)\left(I(\vec{x}+\vec{X})-\mu\right)
```

which can be read as Pearson's correlation coefficient over the set of pixel
pairs that are $`\vec{X}`$ apart. By construction
$`\mathrm{ACF}(\vec{0})=1`$, and $`\mathrm{ACF}(-\vec{X})=\mathrm{ACF}(\vec{X})`$.
Note that $`\mu`$ and $`\sigma^2`$ are those of the whole leaf, and not of the
subset of pixels that contributes at $`\vec{X}`$; this is why
$`\mathrm{ACF}(\vec{X})>1`$ is possible (see technical note 1).

Both sums above are evaluated as cross-correlations using an FFT
(`scipy.signal.correlate`). Note that division by $`n(\vec{X})`$ is required
because the leaf occupies only part of the image: without it, the decay of the
ACF would largely reflect the shrinking overlap of the leaf with its shifted
copy, i.e. the shape of the leaf, rather than the damage pattern. For large
$`\vec{X}`$ few pixel pairs remain and the estimate becomes unreliable, so we
only evaluate displacements with a sufficient number of contributing pairs,

```math
V = \left\{ \vec{X} \; : \; n(\vec{X}) \geq \max(f N, 1) \right\}
\qquad \text{with} \qquad f = 0.05
```

Finally, $`\mathrm{ACF}(d)`$ (the thick black line in the plots above) is
obtained by averaging over all displacements of the same length, where lengths
are binned to integer pixel distances,

```math
\mathrm{ACF}(d) = \frac{1}{|V_d|} \sum_{\vec{X} \in V_d} \mathrm{ACF}(\vec{X})
\qquad \text{with} \qquad
V_d = \left\{ \vec{X} \in V \; : \; \left\lfloor |\vec{X}| \right\rfloor = d \right\}
```
