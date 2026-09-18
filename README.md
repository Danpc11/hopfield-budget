# The Hopfield barrier under a finite kinetic budget

Certified convex optimisation for the dissipation-function design problem in
proofreading networks, together with a closed-form saturation law and its check.

## Problem

For an $m$-stage proofreading network with discrimination factor $F$, we compute
the lowest error reachable when the activity budget is finite and a throughput
$J_0$ is required:

$$\min\ \sigma \quad\text{s.t.}\quad \text{error}\le\varepsilon,\quad J_{\rm prod}\ge J_0,\quad \textstyle\sum_e (f^+_e+f^-_e)\le A$$

with $\sigma=\sum_e (f^+_e-f^-_e)\ln(f^+_e/f^-_e)$, jointly convex in the one-way
fluxes. The shared-rate constraint between substrates is bilinear, so the global
problem is solved by spatial branch and bound over McCormick envelopes, with
upper bounds from rate-space sampling.

## Install

```bash
git clone https://github.com/Danpc11/hopfield-budget.git
cd hopfield-budget
./setup.sh
```

`setup.sh` builds the venv, runs the tests, and prints the verdict for epistasis
and literature, which needs no sweep. If that fails, the environment is wrong.

## Run

Single machine, no scheduler.

```bash
source env.sh
python run_local.py --preset pilot                 # 216 points, pipeline check
python run_local.py --preset coarse --jobs 16      # stage 1: locate the wall
python run_local.py --preset refine --jobs 16      # stage 2: fine grid around it
python verdict.py --results results
python aggregate.py --results results --csv summary.csv
```

`--jobs` defaults to all cores minus one. Ctrl-C is safe: finished points are on
disk and the same command resumes. Writing is atomic. One point costs about 2 s
at 30 nodes and about 2 min at 480 nodes.

Each solve is single threaded and the pool provides the parallelism, so
`run_local.py` sets `OMP_NUM_THREADS=1` before importing numpy.

`make_tasks.py` writes the same grid as a text file for use with `xargs -P` or
another tool, at the cost of one Python start per point.

## Two-stage grid

The eps grid must resolve the wall better than 1/5 of the excess being measured.
At small $J_0$ that excess is about 2%, requiring a log step of 0.004; a uniform
grid over two decades gives 0.14. With a coarse grid the walls are quantised and
the monotonicity and convergence checks pass for the wrong reason. Criterion
**S0** rejects such a grid. The coupling between stages is only in task
generation; every point stays independent and restartable.

## Closed-form law

The saturation curve is solved analytically for $m=2$, with no free parameter.
Apply the matrix-tree theorem to the triangle with the optimal architecture
($a_3=0$, no direct binding to the checked state; $s=b_2/b_3=0$, no
back-stepping). With $r=b_1/a_2$ the rejection ratio at the first checkpoint,

$$\varepsilon(r)=\frac{1+1/r}{F\,(F+1/r)},\qquad v_1(r)=\frac{1+r}{1+Fr}$$

giving $1/F^2$ as $r\to\infty$ and $1/F$ as $r\to 0$. The current is the same on
the three edges of the cycle, so the traffic per edge follows, and the wrong
branch is not negligible: its binding flux is identical and its dissociation flux
is $v_1\pi_1Fb_1$, which equals the right-branch one for large $r$ because
$v_1\to 1/F$. Hence

$$L(r)=\frac{A}{J}=(3+2r)+(1+r)+rFv_1(r)+v_1(r)+\varepsilon(r)F$$

Since the budget is saturated, $L=A/J_0$ is fixed by the operating point: invert
$L(r)=A/J_0$ and read $\varepsilon(r)$.

At collapse ($r\to0$) every edge of both branches carries exactly $J$, so
$L_{\min}=2(m+1)$ and

$$J_c=\frac{A}{2(m+1)}\qquad\Longrightarrow\qquad \frac{J_c(m{=}3)}{J_c(m{=}2)}=\frac{6}{8}=0.75$$

exact and independent of $F$. This is the prediction linking saturation to
topology, and it is the content of criterion S5.

Check against the $m=2$, $F=50$ sweep (5 points, $J_0$ from 0.005 to 0.13):
residuals $+0.01\%$, $-0.09\%$, $-0.79\%$, $-1.22\%$, $-0.84\%$. All within
1.3% with nothing fitted. The residuals are systematically negative by about 1%,
matching the search bias measured in the convergence study: a branch and bound
that misses an incumbent reports the wall too high. The law is the true wall and
the numerics sit just above it.

The sweep therefore tests a parameter-free prediction. A1 checks the size of the
residuals, A2 checks their sign.

### Signomial route

The flux formulation is convex, but the shared-rate constraint is bilinear, and
at small $v$ the positive cone becomes empty in the directions the branch and
bound explores. At $m=3$ that made every point come out "undetermined", which is
a search failure and not physics.

`hopfield/sigp.py` removes $v$ from the problem. By the matrix-tree theorem every
stationary weight is a **posynomial** in the rates, so

$$\varepsilon=\frac{\rho_{W(m)}}{\rho_{R(m)}},\qquad J=\frac{\rho_ik_{ij}-\rho_jk_{ji}}{Z},\qquad A=\frac{\sum_e(\rho_ik_{ij}+\rho_jk_{ji})}{Z}$$

are all ratios of posynomials: a signomial program. The standard solution is
condensation (Duffin, Peterson and Zener 1967; see Chiang's review): replace the
posynomial on the right of each inequality by its monomial lower bound from the
arithmetic-geometric mean inequality at the current point, which makes every step
a geometric program, convex in the log-rates, and iterate. Each subproblem is an
inner approximation, so anything it returns is feasible for the original problem.

Two further points matter in practice. The continuation in $J_0$ must sweep
**both ways**: going up only, the early points keep whatever local optimum the
random starts found while later ones inherit better solutions, and the wall comes
out *decreasing* with $J_0$, which is impossible. And the traffic posynomial has
one term per (edge, spanning tree) pair, about 6000 at $m=3$; merging repeated
exponents brings it to 528 and a single solve from tens of seconds to 1.4 s.

Validation at $m=2$, $F=50$, against the closed-form law: all five points agree
to four decimals (+0.00%) in 8 seconds. The branch and bound needed about 190 s
for three points and sat 1% high. The two methods are independent, so their
agreement is the strongest check we have on the law.

```python
from hopfield.sigp import wall_curve
wall_curve(m=3, F=20.0, J0_list=[3e-3, 1e-2, 2e-2, 4e-2], starts=3, sweeps=2)
```

### Open

At $m=3$ the signomial route runs where the branch and bound returned nothing,
and the curve is monotone, but the values sit 4.8 to 19.7 times above the
theoretical wall $1/F^3$. They are valid upper bounds, not converged ones:
condensation reaches a KKT point, and three starts with two sweeps is not enough.
More starts and more sweeps are now cheap, since one solve costs seconds.

So $J_c \propto 1/(m+1)$, the content of criterion S5, is still unverified. The
route is open and inexpensive; it needs compute, not new ideas.

## Verdict pipeline

`verdict.py` evaluates criteria fixed in advance (`CRITERIA`, not modified after
seeing the data) and prints the falsifiable prediction with its power estimate.

| | decides |
|---|---|
| S0–S5 | is saturation derivable from a finite budget? |
| A1–A2 | does the closed-form law match, with no free parameter? |
| E1–E3 | is epistasis derivable from the shared clock? |
| L1–L2 | does it agree with published data? |

Without a sweep: E1 passes ($|E-1| = 1.1\times10^{-16}$, fixed window), E2 passes
(0 negative cases in 20,000 draws), E3 passes, L1 passes ($E=3.75$, Morrison
1993), L2 passes. Saturation reports NO DATA until the sweep runs.

## Conventions enforced in code

1. **Three-valued status.** An infeasible root relaxation proves impossibility; a
   branch and bound with no incumbent does not. `UNDETERMINED` never counts as
   infeasible, since that inflates the wall and creates spurious transitions.
2. **The wall is an interval.** It is bracketed between the largest eps proved
   infeasible and the smallest feasible eps. The proved side is rare, so the
   usable result is an upper bound decreasing with budget.
3. **The node sweep is the convergence check.** The estimate must decrease with
   budget and settle. Where the excess is comparable to the search bias, only the
   limit can be stated, not the shape. `aggregate.py` flags drift above 1%
   between the two largest budgets and fits only with the largest.
4. **No serial bisection.** The eps grid is dense and every point independent;
   the wall is read from the edge afterwards.

Feasibility must also be monotone in $\varepsilon$ and anti-monotone in $J_0$;
`aggregate.py` lists violations rather than averaging them.

## Layout

```
hopfield/core.py       Net (steady state, EPR, currents, exact covariance), convex sigma
hopfield/bb.py         McCormick + spatial branch and bound, three-valued status
hopfield/models.py     m-stage proofreading network + rate-space seeder
hopfield/analytic.py   closed-form saturation law and its check
hopfield/epistasis.py  shared clock, closed-form law, sign constraint
hopfield/literature.py published values with sources (none fitted)
hopfield/runner.py     evaluate one point, write its JSON (atomic, restartable)
run_local.py           full sweep on one machine, with progress and resume
run_point.py           single point, for debugging
make_tasks.py          grid as a text file, for external tools
aggregate.py           walls, monotonicity and convergence checks, fits
verdict.py             criteria fixed in advance, verdict and prediction
tests/                 19 tests of the invariants (~10 s)
env.sh, setup.sh       environment and one-command install
```

## Reproducibility

Every JSON records all parameters, the seed, the machine, and the Python and
NumPy versions. `env.sh` fixes `PYTHONHASHSEED` and limits BLAS to one thread.
`.gitignore` excludes `results/`, `tasks*.txt` and CSV files; archive sweeps
outside git.

## Before publishing

`LICENSE` and `CITATION.cff` contain author placeholders. `env.sh` has a
commented `module load` line. Add the funding acknowledgement.
