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

With $q$ the fraction of flux rejected at the checkpoint, a checking stage
discriminates by $1/(1+q(F-1))$, so with $m-1$ stages

$$\varepsilon(q)=\frac{1}{F}\big[1+q(F-1)\big]^{-(m-1)},\qquad q_{\max}=1-\frac{L J_0}{A}$$

giving $1/F^m$ at zero throughput and $1/F$ when the capacity is exhausted. With
$x = L J_0/A$ the fraction of budget consumed by throughput, the checking stage
is fully lost at $x=1$:

$$J_c=\frac{A}{L}\qquad (J_c\propto 1/L)$$

exact and independent of $F$. `analytic.py` also returns `J_pole`, the pole of
the algebraic form at $x=F/(F-1)$, since a free fit of $1/\beta$ against $1/J$
extrapolates there rather than to $J_c$; the two differ by $1/(F-1)$.

Checked against the $m=2$, $F=50$ sweep: $L = 4.318$ with 5.4% spread and
largest residual 9.7%, no fitted parameter. The monotone drift of $L$
(4.00 to 4.66) is second order: the futile cycle and the productive path share
their first edges, so the exact budget needs $L_s$, $L_r$ and $L_p$ separately.

The sweep therefore tests a parameter-free prediction rather than searching for a
functional form. Criteria A1 and A2 compare against the law, not a free fit.

### Open

$J_c \propto 1/L$ is decided at $m=3$ and is not yet checked. With three outer
dimensions the seeder does not reach the target region: undriven rate sampling
gives $v_m \sim 1/F$ while the wall is at $1/F^m$, so candidates lie inside the
box but violate $v_m \le \varepsilon$ and no incumbent is found. The symptom is
feasibility that is not monotone in $\varepsilon$. `rate_space_seeder` already
injects a cycle affinity scaling as $3m\ln F$ and filters by distance to target;
this is not sufficient at $m=3$ with 40 nodes. The likely fix is to construct
seeds backwards from the target using the matrix-tree theorem.

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
