# Theory

The derivations behind the code, and an honest account of what each one rests
on. Every number quoted here is produced by the repository; the script that
produces it is named. A status table at the end says which results are proven,
which are measured, which are conjectures, and which have been retracted.

Notation follows the code: $m$ discriminating stages, discrimination factor $F$,
activity budget $A$, demanded throughput $J_0$.

---

## 1. Setting

A continuous-time Markov jump process on $n$ states with edge set $E$. For an
edge joining states $i$ and $j$, the one-way fluxes in the steady state are

$$f^{+}_{e}=\pi_i k_{ij},\qquad f^{-}_{e}=\pi_j k_{ji}$$

with $\pi$ the stationary distribution and $k_{ij}$ the rate from $i$ to $j$. The
net current and the traffic on that edge are $j_e=f^{+}_{e}-f^{-}_{e}$ and
$a_e=f^{+}_{e}+f^{-}_{e}$: the part that transports, and the part that churns.
The entropy production rate, in units of $k_B$, is the Schnakenberg sum

$$\sigma=\sum_{e}\bigl(f^{+}_{e}-f^{-}_{e}\bigr)\ln\frac{f^{+}_{e}}{f^{-}_{e}}$$

`hopfield/core.py` implements all of this exactly, including the long-time
current covariance through the reduced Drazin inverse rather than finite
differences.

---

## 2. Entropy production is convex in one-way fluxes

**Status: proven.**

Each term of $\sigma$ has the form $(x-y)\ln(x/y)$, the symmetrised relative
entropy of the pair. Relative entropy is jointly convex, so $\sigma$ is jointly
convex in $(\mathbf f^{+},\mathbf f^{-})$. This holds for any edge set; nothing
about the network enters.

The same function has a second form that shows what it is. With
$\tilde a_e=2\sqrt{f^{+}_{e}f^{-}_{e}}$ the geometric traffic and
$\xi_e=\ln(f^{+}_{e}/f^{-}_{e})$ the force on the edge,

$$\sigma=\sum_{e}\tilde a_e\,\varphi\!\left(\frac{j_e}{\tilde a_e}\right),
\qquad \varphi(s)=2s\,\mathrm{arcsinh}(s)$$

$\varphi$ is convex and even, so each term is the perspective of a convex
function and therefore jointly convex in $(j_e,\tilde a_e)$. The flux-force
relation implied by the same variables,

$$j_e=\tilde a_e\sinh(\xi_e/2)$$

is the hyperbolic-cosine gradient structure that Mielke, Peletier and Renger
derived for Markov jump processes from the large-deviation rate functional. So
the object being minimised is not an ad hoc cost: it is the dissipation
potential of the process.

Checked to $10^{-15}$ on random flux vectors, and joint convexity by 4000
midpoint tests, in `test/test_core.py`.

### Topology selection is free

**Status: proven, and checked against enumeration.**

Switching an edge off means $f^{+}_{e}=f^{-}_{e}=0$, which lies inside the
feasible set. So optimising over a complete candidate graph optimises over all
of its subgraphs at once. On $K_5$ with the throughput imposed between two fixed
nodes, the convex optimum agrees with the best of the 466 connected subgraphs to
$7\times10^{-9}$, and leaves seven of the ten edges active.

---

## 3. Two substrates: the bilinearity is intrinsic

**Status: proven for the two natural charts.**

Discrimination needs one machine and two substrates. Let the network carry a
right branch and a wrong branch sharing the free-enzyme state, with identical
rates except that every dissociation of the wrong substrate is faster by $F$. In
flux variables,

$$f^{+}_{e,W}=v_{i}\,f^{+}_{e,R},\qquad
f^{-}_{e,W}=v_{j}\,F^{\delta_e}\,f^{-}_{e,R}$$

with $v_k=\pi_{W(k)}/\pi_{R(k)}$ the population ratio at bound state $k$, and
$\delta_e=1$ on edges that dissociate into the free enzyme. The error is exactly
$\varepsilon=v_m$, so the functional requirement is **linear** in $\mathbf v$.
The sharing constraint is **bilinear** in $(\mathbf v,\mathbf f)$.

Moving to logarithmic flux variables makes the sharing constraint affine, and
destroys the convexity of $\sigma$: the smallest Hessian eigenvalue is $-121$ on
the modest box $[-2,2]^{2}$ per edge, and grows in magnitude with the box.

Between the two natural coordinate systems there is none in which the
dissipation functional and the sharing constraint are simultaneously convex. The
bilinearity is a property of "one machine, two substrates", not of the writing.

---

## 4. Certified global optima

**Status: method, standard.**

The objective is convex and only the coupling is bilinear, which suits spatial
branch and bound with McCormick envelopes. For $w=v_k f$ on a box, the four
McCormick inequalities give the convex hull of the product; they are exact at
the corners and relaxed inside, so they tighten as the box shrinks. Replacing
each product by its envelope leaves a convex program whose value lower-bounds
the true optimum on that box. Upper bounds come from fixing $\mathbf v$, which
makes the products affine.

Two points of practice, both of which produced wrong results before being handled
explicitly:

**The node status is three-valued.** An infeasible root relaxation proves
impossibility. A branch and bound that finds no incumbent proves nothing.
Treating the second as the first inflates the error floor and manufactures
transitions that do not exist.

**Missing an incumbent always biases the floor upward**, so the estimate must
decrease with the node budget and settle. Measured directly: at two
representative throughputs the floor fell by 3.2% and 2.3% from 15 to 30 nodes,
and by a further 0.8% from 30 to 60. That sets the residual bias at about one
per cent, and it matters later.

---

## 5. Closed form for $m=2$

**Status: derived, and matched by two independent numerical routes.**

### Stationary weights

By the matrix-tree theorem the unnormalised weight of state $i$ is

$$\rho_i=\sum_{T\in\mathcal{T}}\ \prod_{(u\to w)\in T_{\to i}}k_{uw}$$

summed over spanning trees of the undirected graph, each oriented towards $i$.
Every $\rho_i$ is a posynomial in the rates, and $\pi_i=\rho_i/\sum_l\rho_l$.
Verified against the exact steady state to $10^{-12}$ for $m=2$ and $m=3$.

### The error

Take $m=2$, so each branch is a triangle on the free enzyme and two bound
states. The architecture that minimises the error has no direct binding to the
checked state and no back-stepping, which one can read off the tree sums term by
term. With $r=b_1/a_2$ the ratio of the rejection rate at the first checkpoint to
the forward rate out of it,

$$\varepsilon(r)=\frac{1+1/r}{F\,(F+1/r)},\qquad
v_{1}(r)=\frac{1+r}{1+Fr}$$

$\varepsilon\to F^{-2}$ as $r\to\infty$, and $\varepsilon\to F^{-1}$ as $r\to0$.

### The traffic

The net current is the same on the three edges of the cycle, so the traffic per
edge follows from $\pi_2 b_3=J$ and $\pi_1 a_2=J$:

| | $(0,2)$ | $(1,2)$ | $(0,1)$ |
|---|---|---|---|
| right branch | $J$ | $J$ | $J(1+2r)$ |
| wrong branch | $\varepsilon F J$ | $v_1 J$ | $J(1+r)+JrFv_1$ |

The wrong branch is not negligible. Its binding flux is identical to the right
branch, since the two share both the rate and the source state, and its
dissociation flux is $v_1\pi_1 F b_1$, which equals the right-branch value
whenever $v_1\to1/F$. The enzyme spends as much turnover rejecting wrong
substrates as processing right ones, which is what proofreading is. Collecting
the six contributions,

$$L(r)\equiv\frac{A}{J}=(3+2r)+(1+r)+rFv_{1}(r)+v_{1}(r)+\varepsilon(r)F$$

Because the budget is saturated at the optimum, $L=A/J_0$ is fixed by the
operating point. Invert $L(r)=A/J_0$ for $r$, substitute into $\varepsilon(r)$,
and the error floor follows with **no adjustable parameter**.

### Collapse: $L(r)$ is not monotone

Setting $r=0$ gives $L=2(m+1)=6$, twice the cycle length, because both branches
then carry the full current on every edge. That is **not** the minimum.

Using $\varepsilon(r)F=v_1(r)$, which follows from the two expressions above, and
writing $u=Fr$,

$$L(u)=4+3r+(1+r)\frac{u+2}{1+u}\;\simeq\;5+\frac{1}{u}+\frac{4u}{F}$$

the second form for $u\gg1$ and to leading order in $1/F$. It has an interior
minimum at $u^{*}=\sqrt{F}/2$:

$$r^{*}=\frac{1}{2\sqrt{F}},\qquad
L_{\min}=5+\frac{4}{\sqrt{F}}+O(F^{-1}),\qquad
J_c=\frac{A}{L_{\min}}$$

A small rejection ratio suppresses the wrong branch, since $v_1<1$, and the
turnover saved there exceeds the cost of the futile cycle. **Proofreading is not
switched off at collapse; it is held at $r^{*}$.** For $F=50$ the effective
number of stages therefore stops at $n_{\rm eff}=1.308$, not at 1.

| $F$ | $r^{*}$ | $1/(2\sqrt F)$ | $L_{\min}$ | $5+4/\sqrt F$ | $J_c$ |
|---|---|---|---|---|---|
| 20 | 0.0590 | 0.1118 | 5.7218 | 5.8944 | 0.1748 |
| 50 | 0.0500 | 0.0707 | 5.5000 | 5.5657 | 0.1818 |
| 200 | 0.0303 | 0.0354 | 5.2671 | 5.2828 | 0.1899 |
| $10^{4}$ | 0.00490 | 0.00500 | 5.0397 | 5.0400 | 0.1984 |

The asymptotic form is accurate to 1.2% at $F=50$ and 0.1% at $F=10^{4}$; the
code uses the numerical minimum through `r_star()`. At $F=50$, $A=1$ it gives
$J_c=0.1818$, against $0.182$ from an independent free fit of the numerical
curve made before the derivation existed.

### What was retracted

An earlier version of this document and of the README claimed

$$J_c=\frac{A}{2(m+1)}\quad\text{"exact and independent of }F\text{"},\qquad
\frac{J_c(m{=}3)}{J_c(m{=}2)}=\frac{6}{8}=0.75$$

That is wrong. It used $L(0)$ instead of $\min_r L(r)$, which puts the collapse
point about 5% too low at $F=20$ and 9% too low at $F=50$. The ratio 0.75 was a
consequence of the same error and does not survive it. There is currently **no**
prediction for how $J_c$ scales with $m$: the traffic count of the table above
has to be redone stage by stage, and that has not been done.

`run_sigp.py` aborts if it finds an `analytic.py` without `r_star`, and the CI
asserts $L_{\min}(F)<6$, so this particular mistake cannot come back silently.

### Agreement with the numerics

Against the certified branch and bound at $m=2$, $F=50$, with nothing fitted:

| $J_0$ | law | numerics | residual |
|---|---|---|---|
| $5\times10^{-3}$ | 1.0201 | 1.020 | $+0.01\%$ |
| $2\times10^{-2}$ | 1.0870 | 1.088 | $-0.09\%$ |
| $4\times10^{-2}$ | 1.1955 | 1.205 | $-0.79\%$ |
| $8\times10^{-2}$ | 1.5192 | 1.538 | $-1.22\%$ |
| $1.3\times10^{-1}$ | 2.4393 | 2.460 | $-0.84\%$ |

All within 1.3%. The residuals are systematically **negative** by about one per
cent, which is exactly the search bias measured in section 4: a branch and bound
that misses an incumbent reports the wall too high. The law is the true wall and
the numerics sit just above it. That sign is checked as a test, not just
observed.

---

## 6. The signomial route

**Status: method, standard; reproduces the law to four decimals.**

The flux formulation carries a bilinearity, and at small $\mathbf v$ the positive
cone becomes empty in the directions the branch and bound explores. At $m=3$ that
made every point come out undetermined, which is a search failure and not
physics.

The matrix-tree theorem removes $\mathbf v$ from the problem. In terms of the
rates, the error, the throughput and the traffic are all ratios of posynomials,

$$\varepsilon=\frac{\rho_{W(m)}}{\rho_{R(m)}},\quad
J=\frac{\rho_ik_{ij}-\rho_jk_{ji}}{Z},\quad
A=\frac{\sum_e(\rho_ik_{ij}+\rho_jk_{ji})}{Z},\qquad Z=\sum_l\rho_l$$

so the problem is a signomial program, and the classical way to solve one is
condensation. At a point $x_0$ in log-rate coordinates, a posynomial
$p(x)=\sum_k c_k e^{\mathbf a_k\cdot x}$ admits the monomial lower bound

$$\ln\hat p(x)=\ln p(x_{0})+\Bigl(\textstyle\sum_k u_k\mathbf a_k\Bigr)\cdot(x-x_{0}),
\qquad u_k=\frac{c_ke^{\mathbf a_k\cdot x_{0}}}{p(x_{0})}$$

from the arithmetic-geometric mean inequality, tight at $x_0$. Replacing the
posynomial on the right of each inequality by $\hat p$ makes every step a
geometric program, convex in the log-rates, and any solution it returns is
feasible for the original problem.

Four implementation points, each of which was a failure before it was a fix:

**Continuation must sweep both ways.** Going up in $J_0$ only, the early points
keep whatever local optimum the random starts found while later ones inherit
better solutions, and the computed floor *decreases* with throughput, which is
impossible.

**An auxiliary ladder is needed near collapse.** Random starts almost never land
in the feasible set there, so asking only for throughputs near $J_c$ leaves the
first phase with nothing and the chains with no seed. The ladder is spaced in
$\log(J_c-J_0)$, not in $\log J_0$: near the pole the error floor depends on the
distance to collapse, and a grid geometric in $J_0$ puts its largest step exactly
where the problem is most sensitive.

**Merge repeated exponents.** The traffic posynomial has one term per (edge,
spanning tree) pair, about 6000 at $m=3$; merging brings it to 528 and one solve
from tens of seconds to 1.4 s.

**Work in log space.** Evaluating $\sum_k c_k e^{\mathbf a_k\cdot x}$ directly
overflows at $m=4$, where the tree monomials reach degree four over log-rates of
order nine. `log_value` uses log-sum-exp and condensation computes its weights as
$\exp(\ln c_k+\mathbf a_k\cdot x_0-\ln p)$.

Validation at $m=2$: 23 points at $F=50$ and 30 at $F=20$, all monotone, agreeing
with the closed-form law to 0.00–0.08%, reaching 92% of the collapse throughput
where $\varepsilon$ has already risen fourfold over the Hopfield barrier. Two
methods sharing only the physical model, agreeing to four decimals, is the
strongest check available on section 5.

---

## 7. Epistasis as a second-moment effect

**Status: identity exact; the biological identification is a hypothesis.**

Two repair systems in series are usually assumed to multiply. With $\alpha$ and
$\beta$ the probabilities that an error escapes the first and the second, the
deviation from multiplicativity is

$$\mathcal{E}\equiv\frac{\mu_{0}\,\mu_{AB}}{\mu_{A}\,\mu_{B}}
=\frac{\langle\alpha\beta\rangle}{\langle\alpha\rangle\langle\beta\rangle}
=1+\frac{\mathrm{Cov}(\alpha,\beta)}{\langle\alpha\rangle\langle\beta\rangle}$$

where $\mu_0,\mu_A,\mu_B,\mu_{AB}$ are the error rates with neither, either and
both systems removed. This is an identity, not a model.

If the two systems race against a common clock, the heterogeneity of that clock
generates the covariance by itself. With $\gamma$ the rate at which the shared
window closes and $r_1,r_2$ the two repair rates,
$\alpha=\gamma/(\gamma+r_1)$ and $\beta=\gamma/(\gamma+r_2)$, and to second order

$$\mathcal{E}-1\simeq\mathrm{CV}^{2}(\gamma)\,(1-\alpha)(1-\beta)$$

Three consequences, all tested:

- A **fixed** window gives $\mathcal{E}=1$ to machine precision
  ($|\mathcal E-1|=1.1\times10^{-16}$): multiplicativity is exact.
- A shared clock can only give $\mathcal{E}>1$. Over 20 000 random draws the
  smallest value found was $1.000001$, none negative. The mechanism is therefore
  more constrained, and more falsifiable, than an account based on overlapping
  specificities, which admits either sign.
- The closed form matches the measurement to within 15.3% over random parameters.

In statistics this is a **shared frailty** effect, standard in survival analysis.
What is specific here is the identification of the shared clock with the finite
kinetic window of the replication fork, for which there is direct single-cell
evidence. That identification is a hypothesis, not a derivation.

### Comparison with published data

Morrison et al. (1993), yeast, same reporter: proofreading-deficient 130,
mismatch-repair-deficient 41, double $2\times10^{4}$, giving
$\mathcal{E}=3.75$. Super-multiplicative, consistent with the sign constraint.

Tumour mutational burden gives $\mathcal{E}\approx0.14$–$0.28$,
sub-multiplicative. That does **not** falsify the model: the predicted effect
along the replication-timing axis is 1.2–4.9%, while the observed between-genotype
deviation is 86%, seventeen times larger and of the opposite sign. TMB is an
accumulated burden censored from above by error-induced extinction, so it
measures censoring rather than repair kinetics. `verdict.py` checks this
explicitly as criterion L2.

### The falsifiable prediction

$$\mathcal{E}-1=\rho\,\mathrm{CV}(\alpha)\,\mathrm{CV}(\beta)=0.012\ \text{to}\ 0.049$$

with both coefficients of variation taken from published replication-timing
gradients and $\rho\le1$; no free parameter. Falsifiable twice over: by sign,
since a shared clock cannot give $\mathcal E<1$; and by size. It must be measured
**inside** one genome, resolved by replication timing, in ultramutated
POLE+MMRd tumours, and it needs whole genomes rather than exomes, because exons
sit in early-replicating open chromatin and squeeze the very axis being measured.

---

## 8. Relation to the force bounds

Owen, Gingrich and Horowitz bound the discriminatory index of a proofreading
scheme by the cycle affinity, $|\nu-1|\le(m-1)\tanh(F_{E\leftrightarrow ES}/4)$,
so that enough driving recovers all $m$ stages. That bound is on the
**force**. Section 5 says the recovery is unreachable at finite throughput,
however large the driving, because the constraint is on the **activity**.

On a single-cycle graph the two statements are the two factors of the same
product, since Schnakenberg's formula reduces to $\sigma=J\,\Delta\mu$: the
published bound constrains $\Delta\mu$, and ours constrains what the pair
$(J,\Delta\mu)$ can be at fixed activity. They are companions, not competitors.

---

## 9. Status

| Result | Status | Evidence |
|---|---|---|
| $\sigma$ jointly convex in one-way fluxes | proven | identity; 4000 midpoint tests |
| Perspective and cosh forms | proven | verified to $10^{-15}$ |
| Topology selection free | proven | $K_5$ against 466 subgraphs, $7\times10^{-9}$ |
| Bilinearity intrinsic to two charts | proven | Hessian eigenvalue $-121$ |
| $\varepsilon(r)$, $v_1(r)$, $L(r)$ for $m=2$ | derived | matrix-tree, term by term |
| $r^{*}=1/(2\sqrt F)$, $L_{\min}=5+4/\sqrt F$ | derived | asymptotic; numerical minimum in code |
| $J_c=A/L_{\min}$ | derived | $0.1818$ at $F=50$ vs $0.182$ free fit |
| Law matches numerics, no free parameter | measured | 53 points, two $F$, 0.00–1.3% |
| Residual bias is the search bias | measured | sign negative, magnitude 1% |
| Epistasis identity and sign constraint | proven | 20 000 draws, none negative |
| Shared clock is the replication window | hypothesis | single-cell evidence, not derived |
| $m=3$ wall approaches $1/F^{3}$ | **open** | 1.075 at $J_0=10^{-4}$; floor not identified |
| Exponent of the approach, $m\ge3$ | **open** | fit not identifiable over 1.5 decades |
| $J_c$ versus $m$ | **open** | traffic count not redone for $m>2$ |
| $J_c=A/2(m+1)$, ratio $0.75$ | **retracted** | used $L(0)$, not $\min_r L(r)$ |

### The three open items, concretely

**Does the $m=3$ wall reach $1/F^{3}$?** Over $J_0$ from $10^{-4}$ to
$3\times10^{-3}$ the ratio falls only from 1.075 to 1.266. A fit with a free
floor is not identifiable over that range: applied to $m=2$, where the law
requires a floor of exactly 1, the same fit returns 0.90. Needs two decades
lower. If the ratio flattens above 1, the Hopfield limit would be unreachable for
three stages even at zero throughput, which would be a stronger statement than
any exponent.

**Where is $J_c(m{=}3)$?** The grid has not reached close enough to the
divergence; a fit for the pole returns the edge of its search range.

**How does $J_c$ scale with $m$?** Not derived and not measured. This is the
content of criterion S5 in `verdict.py`, which reports NO DATA.

One convergence point is settled: at $J_0=2.083\times10^{-3}$ the $m=3$ wall is
1.2089 with 48 starts and 1.2089 with 300, so that value is converged and the
earlier figure of 4.8, obtained with three starts, was a search failure.
