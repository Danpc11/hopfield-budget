# The Hopfield barrier under a finite kinetic budget

This code computes the lowest error $\varepsilon_{\min}(J_0)$ that an $m$-stage
proofreading network can reach, when the activity budget (total traffic) is
finite and the network must also deliver a throughput $J_0$. The discrimination
factor is $F$.

## What it computes

At every grid point it solves

$$\min\ \sigma \quad\text{subject to}\quad \text{error}\le\varepsilon,\quad J_{\rm prod}\ge J_0,\quad \textstyle\sum_e (f^+_e+f^-_e)\le A$$

with $\sigma=\sum_e (f^+_e-f^-_e)\ln(f^+_e/f^-_e)$, which is **jointly convex**
in the one-way fluxes. The two substrates must share the same rates, and that
constraint is bilinear. So the global problem is solved with spatial branch and
bound over McCormick envelopes. Upper bounds come from sampling in rate space.

## Install

```bash
git clone https://github.com/Danpc11/hopfield-budget.git
cd hopfield-budget
./setup.sh          # builds the venv, installs, runs the tests and the verdict
```

`setup.sh` is safe to run again. It ends by printing the verdict for epistasis
and literature, which needs no sweep. **If that part does not come out clean, the
environment is wrong and there is no point in starting a sweep.**

### First push

```bash
git init -b main
git add .
git commit -m "The Hopfield barrier under a finite kinetic budget"
git remote add origin git@github.com:Danpc11/hopfield-budget.git
git push -u origin main
```

`.gitignore` excludes `results/`, `tasks*.txt` and CSV files. The sweep writes
tens of thousands of JSON files, and they must not go into git. To archive
results, compress them and upload to Zenodo or to your institute storage.

### Tests

```bash
pytest -q          # 16 tests, about 6 seconds
```

They cover the exact identities ($\sigma = J\cdot A$, the rank of the covariance
equal to the cycle rank, the perspective form, the cosh relation), joint
convexity, the sign constraint of the epistasis, the closed-form law, and that
the branch-and-bound status is still **three-valued**. Each test matches one
check that, when it broke during development, produced a wrong result.
GitHub Actions runs them on Python 3.10 and 3.12.

## Run

Everything runs on one machine. There is no scheduler and no job script.

```bash
source env.sh

# 0) check the pipeline (216 points, a few minutes)
python run_local.py --preset pilot
python verdict.py --results results

# 1) coarse stage: find the wall
python run_local.py --preset coarse --jobs 16

# 2) fine stage: dense grid around the wall found in stage 1
python run_local.py --preset refine --jobs 16

# 3) verdict
python verdict.py --results results
python aggregate.py --results results --csv summary.csv
```

`--jobs` defaults to all cores minus one. `run_local.py` prints progress and an
estimated time left.

**You can stop it with Ctrl-C at any time.** Nothing is lost: every finished
point is already on disk, and running the same command again continues where it
stopped. Writing is atomic (write to a temporary file, then rename), so a killed
run never leaves a half-written file.

Cost: one point takes about 2 seconds with a budget of 30 nodes, and about
2 minutes with 480 nodes. On 16 cores the coarse stage takes a few hours.

Each solve is single threaded on purpose and the pool gives the parallelism.
`run_local.py` sets `OMP_NUM_THREADS=1` before importing numpy. If you let BLAS
open its own threads, the workers fight for the same cores and everything gets
slower.

If you prefer another tool, `make_tasks.py` writes the same grid as a text file:

```bash
python make_tasks.py --preset coarse > tasks.txt
cat tasks.txt | xargs -P 16 -I{} sh -c "{}"
```

This is slower, because it starts a new Python process for every point.

## Why two stages

The eps grid must resolve the wall better than **1/5 of the excess we want to
measure**. At small $J_0$ that excess is about 2%, so we need a log step of
0.004. A uniform grid over two orders of magnitude gives 0.14, which is thirty
times too coarse. With such a grid the walls are **quantised**, and then the
monotonicity and convergence checks would pass for the wrong reason.

Criterion **S0** exists to catch exactly that, and it was tested against a grid
that was too coarse: it rejected it. The coupling between the two stages is only
in how tasks are generated. Every point is still independent and restartable.

## The closed-form law

You do not need the cluster to find the shape of the saturation curve. You can
derive it. If $q$ is the fraction of flux rejected at the checkpoint, that stage
discriminates by $1/(1+q(F-1))$, and with $m-1$ stages

$$\varepsilon(q)=\frac{1}{F}\big[1+q(F-1)\big]^{-(m-1)},\qquad q_{\max}=1-\frac{L J_0}{A}$$

This gives $1/F^m$ when no throughput is asked, and $1/F$ when the capacity is
used up. That is the gradual loss of stages, derived rather than fitted.

Write $x = L J_0 / A$: the fraction of the traffic budget that the throughput
itself uses up. Three points are easy to confuse:

| $x$ | what happens |
|---|---|
| $1$ | $\varepsilon = 1/F$. The checking stage is completely lost. **This is the physical collapse point.** |
| $1 + 1/F$ | $\varepsilon = 1$. No discrimination is left at all. |
| $1 + \frac{1}{F-1}$ | the denominator vanishes: the pole of the algebraic form, already outside the physical range. |

They are separated by terms of order $1/F$, because
$\frac{F}{F-1} = 1 + \frac1F + \frac1{F^2} + \cdots$, so for large $F$ they sit
almost on top of each other. Note that $1/F$ is also the error one
discrimination stage can reach, so the small parameter of the expansion is the
same number that sets the physics.

We therefore report the physical point:

$$J_c=\frac{A}{L}\qquad (J_c\propto 1/L)$$

which is exact and does not depend on $F$. `analytic.py` also returns `J_pole`,
because a free fit of $1/\beta$ against $1/J$ extrapolates to the pole and not
to $J_c$. The two differ by about $1/F$: 2% at $F=50$, 5% at $F=20$.

`hopfield/analytic.py` checks this against the $m=2$, $F=50$ sweep:
**L = 4.318 with 5.4% spread and a largest residual of 9.7%, with no fitted
parameter.** The slow drift of $L$ (4.00 to 4.66) is a second order effect: the
futile cycle and the productive path share their first edges, so the exact budget
needs $L_s$, $L_r$ and $L_p$ separately.

**This changes the purpose of the sweep**: it no longer has to find the shape, it
has to *check a prediction with no free parameter*. Criteria A1 and A2 test
against the formula, not against a free fit.

### What is still missing, and why

$J_c \propto 1/L$ is decided at $m=3$, and that is **not checked yet**. The
problem is understood. With three outer dimensions the seeder does not reach the
target region. Sampling rates without driving gives $v_m \sim 1/F$, while the
wall is at $1/F^m$. The candidates fall inside the box but break
$v_m \le \varepsilon$, so the branch and bound finds no incumbent. The symptom is
feasibility that is **not monotone** in $\varepsilon$, which is physically
impossible, and that is what we saw.

`rate_space_seeder` already adds a cycle affinity that scales as $3m\ln F$ and
filters by distance to the target. **That was not enough**: at $m=3$ with 40
nodes everything is still undetermined. This is the first problem to solve on the
cluster. The likely fix is to build the seeds by solving backwards from the
target, using the matrix-tree theorem, instead of sampling and hoping.

## The verdict pipeline

`verdict.py` answers three questions using criteria that are **written down
before** looking at the data (in `CRITERIA`, and we do not change them after):

| | what it decides |
|---|---|
| S0-S5 | can we derive **saturation** from a finite budget? |
| A1-A2 | does the **closed-form law** match the sweep, with no free parameter? |
| E1-E3 | can we derive **epistasis** from the shared clock? |
| L1-L2 | does it agree with published data? |

Then it prints the **falsifiable prediction** with its power calculation.

Status without a sweep (epistasis and literature only), already checked:
E1 passes ($|E-1| = 1.1\times10^{-16}$ with a fixed window), E2 passes (0 negative
cases in 20,000 draws), E3 passes, L1 passes ($E=3.75$ in Morrison 1993), L2
passes. **Saturation stays as NO DATA until the sweep runs**: that is what the
cluster is for.

## The four traps this code avoids

Each one produced a false result during manual work, so each one is now written
into the code.

**1. Infeasible is not the same as not found.** The status has three values. An
infeasible root relaxation is a **proof** of impossibility. A branch and bound
that finds no incumbent **is not**. Mixing them makes the wall look higher and
invents transitions that do not exist. `UNDETERMINED` never counts as infeasible.

**2. The wall is an interval, not a number.** It is bracketed between the largest
eps proved infeasible and the smallest feasible eps. The grey zone between them
is reported, not filled in. In practice the "proved infeasible" side is rare,
because the McCormick relaxation seldom closes below the wall. So the useful
result is an **upper bound that goes down as the budget grows**.

**3. The node sweep is not optional.** Missing an incumbent makes the wall look
higher, so the estimate must **go down** with the budget and then settle. If the
excess we measure is about the same size as the search bias, which happens at
small $J_0$ where the excess is 2% and the bias is 1%, then we can only state the
limit, **not the shape**. `aggregate.py` marks a series as NOT CONVERGED when the
drift between the two largest budgets is above 1%, and it fits the shape only
with the largest budget.

**4. No serial bisection.** Bisection carries the error of one point into all the
following ones, and it does not parallelise. Here the eps grid is dense, every
point is independent, and the wall is read afterwards from the edge.

There is one more check: feasibility must be monotone in $\varepsilon$ and go the
other way in $J_0$. `aggregate.py` lists any violation instead of averaging it
away.

## Layout

```
hopfield/core.py       Net (steady state, EPR, currents, exact covariance), convex sigma
hopfield/bb.py         McCormick + spatial branch and bound, three-valued status
hopfield/models.py     m-stage proofreading network + rate-space seeder
hopfield/analytic.py   closed-form saturation law and its check
hopfield/epistasis.py  shared clock, closed-form law and sign constraint
hopfield/literature.py published values with sources (none of them is fitted)
hopfield/runner.py     evaluate one point and write its JSON (atomic, restartable)
run_local.py           run the whole sweep on this machine, with progress and resume
run_point.py           one single point, for debugging
make_tasks.py          write the grid as a text file (only if you want another tool)
aggregate.py           walls, monotonicity and convergence checks, fits
verdict.py             verdict with criteria fixed in advance + prediction
tests/                 16 fast tests of the invariants
env.sh                 venv and variables (single-threaded solver)
setup.sh               one command install and check
```

## Reproducibility

Every JSON stores all the parameters, the seed, the machine name, and the Python
and NumPy versions. `env.sh` fixes `PYTHONHASHSEED` and limits BLAS to one
thread.

## Before you publish the repository

Three things to edit:

1. **`LICENSE`** -- MIT is a placeholder and the holder says `<AUTHORS>`. Check
   what your institute requires about software ownership.
2. **`CITATION.cff`** -- author names are placeholders.
3. **`env.sh`** -- there is a commented line `module load python/3.11`. Adapt it
   to the modules on your cluster.

You should also add the funding acknowledgement.
