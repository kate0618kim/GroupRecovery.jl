# GroupRecovery.jl

`GroupRecovery.jl` is a Julia package for recovering an unknown permutation group G from errored permutation samples.

The package models an error-prone sampler that returns:
- an element of the target group $G$ with probability $1-p$,
- a random element of $S_n$ with error probability $p$.

It provides statistical tests and recovery routines based on the Group Recovery NiAGRA workflow, to recover the underlying group G with arbitrarily high confidence. 

## Features

- `RecoveryDatum`: central mutable state for sampling model, bounds, and discovered properties.
- Statistical distinguishers:
  - `HoeffdingDistinguisher`
  - `explicit_distinguisher`
  - `SPRT_distinguisher`
- Structural Property Recovery Tests:
  - `giant_test`
  - `subgroup_test`
  - `alternating_test`
  - `ktransitivity_test`, `transitivity_test`
  - `orbit_recovery`
  - `find_supergroup`
- Main Group Recovery Algorithm:
  - `NaiveGroupRecovery`
  - `Q_ErrorDetection`
  - `NiAGRA`
  - `main_group_recovery`


## Requirements

- Julia (1.10+ recommended)
- Dependencies (from `Project.toml`):
  - `Oscar`
  - `Distributions`
  - `Plots`
  - `SpecialFunctions`
  - `StatsPlots`

## Installation

From a Julia REPL in this folder (`Code/GroupRecovery`):

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

If you want to use it from another Julia environment, you can `dev` this path:

```julia
using Pkg
Pkg.develop(path="/Users/katekim/Library/CloudStorage/Dropbox/GroupTheoreticRANSAC/Code/GroupRecovery")
```

## Quick start

```julia
using Oscar
using GroupRecovery

# Example target group and noise level
G = alternating_group(10)
p = 0.2

# Build recovery datum from (G, p)
RD = RecoveryDatum(G, p)

# Recover candidate group (adaptive pipeline)
G_est = main_group_recovery(RD; alpha=0.01, method=:hoeffding)

println("Recovered group G_est=G is ", G_est==G)
println("Samples used: ", samples_used(RD))
```

## Core object: `RecoveryDatum`

`RecoveryDatum` stores:
- `X`: sampling function
- `P`: permutation-level properties (filters on sampled permutations)
- `Q`: group-level properties
- `n`: degree
- `p_bound`: upper bound on noise rate $p$
- `order_bound`: upper bound on $|G|$
- `B_bound`: upper bound on constrained sample mass
- `n_samples`: total samples consumed
- `min_supergroup`: current supergroup placeholder

Useful mutators include:
- `reset_sample_count!`, `reset_P!`, `reset_Q!`, `reset_all!`
- `update_p_bound`, `update_order_bound`, `update_B_bound`, `update_all_bounds`
- `add_permutation_property`, `add_group_property`

## Typical workflow

1. Create `RD = RecoveryDatum(G, p)` .
4. Run `NiAGRA` or `main_group_recovery`.
5. Inspect `samples_used(RD)` and resulting group.

## Notes

- Many routines are probabilistic and depend on confidence parameters such as `alpha`.
- Results are sensitive to the quality of `p_bound`, `order_bound`, and `B_bound`.

## Source

Main module file:
- `src/GroupRecovery.jl`
