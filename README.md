
> ### *Under active development*

# *L1DRAC*

[![Build Status](https://github.com/AdityaGahlawat/L1DRAC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/AdityaGahlawat/L1DRAC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/AdityaGahlawat/L1DRAC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/AdityaGahlawat/L1DRAC.jl)

## Table of Contents
- [Description](#descrrption)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Benchmarks](#benchmarks)
- [Example](#example)
- [TODO](#todo)

### Description

Package for the numerical implementation of ***$\mathcal{L}_1$ Distributionally Robust Adaptive Control ($\mathcal{L}_1$-DRAC)***. Find out more about it in our [paper](https://adityagahlawat.github.io/Preprints/DRAC.pdf):
```
@article{gahlawat2025DRAC,
  title={$\mathcal{L}_1$-DRAC: Distributionally Robust Adaptive Control},
  author={Gahlawat, Aditya and Karumanchi, Sambhu and Hovakimyan, Naira},
  journal={arXiv preprint TBD},
  year={2025}
}
```


### Installation
Start Julia with [multi-threading](https://docs.julialang.org/en/v1/manual/multi-threading/) 
```bash
$ julia -t auto  
# or  
$ julia -t <num threads>
```
Add package 
```julia
julia> ] add https://github.com/AdityaGahlawat/L1DRAC.jl
julia> ] Threads.nthreads() # verify num threads
```

### Project Structure

```
L1DRAC/
├── airlock/          # Work-in-progress code (not yet integrated)
├── archive/          # Deprecated/old code for reference
├── benchmark/        # Performance benchmarks
├── examples/
│   └── ex1/
├── sandbox/          # Development scratch work
├── src/              # Main package source
│   ├── L1DRAC.jl           # Main module
│   ├── types.jl            # Type definitions
│   ├── auxiliary.jl        # Backend selection & cleanup functions
│   ├── nominal_system.jl   # NominalSystem simulation (CPU + GPU)
│   ├── true_system.jl      # TrueSystem simulation (CPU + GPU)
│   └── L1_system.jl        # L1-DRAC simulation (CPU + GPU)
├── test/             # Package tests (runtests.jl)
├── writeups/         # Documentation and derivations
├── LICENSE
├── Manifest.toml
├── Project.toml
└── README.md
```

### Benchmarks

![](benchmark/benchmark_results.png)

Computation time benchmark on a simple 2-D system.

**Source:** [`benchmark/benchmark_baseline_Mk3.jl`](benchmark/benchmark_baseline_Mk3.jl)

**Hardware:** Intel i9-10920X (24 threads), 3x NVIDIA RTX A4000 (16GB each)



### Example - STALE (to be edited)

```julia
using L1DRAC
using LinearAlgebra
using Distributions
using ControlSystemsBase

# Simulation Parameters
tspan = (0.0, 5.0)
Δₜ = 1e-4 # Time step size
Ntraj = 1000 # Number of trajectories in ensemble simulation
Δ_saveat = 1e2*Δₜ # Needs to be a integer multiple of Δₜ
simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)


# System Dimensions 
n=2
m=1
d=2
system_dimensions = sys_dims(n, m, d)

# Nominal Vector Fields
λ = 3.0 # Stability of nominal system 
function trck_traj(t) # Reference trajectory for Nominal deterministic system to track 
    return [5*sin(t) + 3*cos(2*t); 0.]
end
function stbl_cntrl(λ) # Stabilizing controller via pole placement
    A = [0 1.0; 0 0]
    B = [0; 1.0] 
    C = I(2)
    D = 0.0 
    sys = ss(A, B, C, D)
    # DesiredPoles = 3*[-2+0.5im, -2-0.5im]
    DesiredPoles = -λ*ones(2)
    K = place(sys, DesiredPoles) # Poles of A-B*K
    return K
end
function f(t,x)
    A = [0 1.0; 0 0]
    B = [0; 1.0]
    K = stbl_cntrl(λ)
    return (A-B*K)*x + B*K*trck_traj(t)
end
g(t) = [0; 1]
g_perp(t) = [1; 0];
p_um(t,x) = 2.0*[0.01 0.1]
p_m(t,x) = 1.0*[0.0 0.8]
p(t,x) = vcat(p_um(t,x), p_m(t,x)) 
nominal_components = nominal_vector_fields(f, g, g_perp, p)

# Uncertain Vector Fields 
Λμ_um(t,x) = 1e-2*(1+sin(x[1]))
Λμ_m(t,x) = 1.0*(5+10*cos(x[2])+5*norm(x))
Λμ(t,x) = vcat(Λμ_um(t,x), Λμ_m(t,x)) 
Λσ_um(t,x) = 1e-2*[0.1+cos(x[2]) 2]
Λσ_m(t,x) = 1.0*[0.0 5+sin(x[2])+5.0*sqrt(norm(x))]
Λσ(t,x) = vcat(Λσ_um(t,x), Λσ_m(t,x))
uncertain_components = uncertain_vector_fields(Λμ, Λσ)

# Initial distributions
nominal_ξ₀ = MvNormal(20.0*ones(n), 1e2*I(n))
true_ξ₀ = MvNormal(-2.0*ones(n), 1e1*I(n))
initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

# L1 DRAC Parameters  
ω = 50.    
Tₛ = 10*Δₜ # Needs to be a integer multiple of Δₜ
λₛ = 100. # Predictor Stability Parameter 
L1params = drac_params(ω, Tₛ, λₛ)

###################################################################
## COMPUTATION 
##################################################################

# Define the systems
nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)


# Solve for Single Sample Paths
nom_sol = system_simulation(simulation_parameters, nominal_system);
tru_sol = system_simulation(simulation_parameters, true_system);
L1_sol = system_simulation(simulation_parameters, true_system, L1params);

# Solve for Ensembles of Ntraj Sample Paths
ens_nom_sol = system_simulation(simulation_parameters, nominal_system; simtype = :ensemble);
ens_tru_sol = system_simulation(simulation_parameters, true_system; simtype = :ensemble);
ens_L1_sol = system_simulation(simulation_parameters, true_system, L1params; simtype = :ensemble);

# Plots
include("plotutils.jl")
plotfunc()
```

![](https://github.com/AdityaGahlawat/L1DRAC.jl/blob/main/examples/ex1/Ex1plot.png)



## TODO
- Parallelized plot utilities (multithreading loops/?)
- baseline control function
- Control logging 
    - Baseline 
    - L1
    - Total
- Parallelized empirical distributions
- Sharper bounds computation
- Manually serialize batches for required mem > available mem on GPUs
- Add ```struct``` wrappers to auto extend necessary function signatures to the complete ```(t,x,dynamics_params)``` for GPU computation. 
    - E.g, ```g(t) -> g(t,x,dynamics_params)```.

---

## Important Changes to Document

#### 1. GPU Parallelization (`max_GPUs` flag)

**What it does:** Controls whether ensemble simulations run on CPU or GPU.

**Values:**
- `max_GPUs = 0` → Force CPU only (uses multi-threading via `EnsembleThreads`)
- `max_GPUs = 1` → Use single GPU if available, else fall back to CPU (default)
- `max_GPUs = N` → Use up to N GPUs (via Distributed.jl, adds overhead)

**How it works:** The package detects available GPUs automatically via `CUDA.devices()`. The actual number used is `numGPUs = min(max_GPUs, available_GPUs)`. So if you set `max_GPUs = 1` but have no GPU, it automatically falls back to CPU.

**Requirements for GPU:** NVIDIA GPU + CUDA drivers + CUDA.jl package. If these aren't present, the package falls back to CPU.

**Why single GPU is recommended:** Multi-GPU uses Distributed.jl which spawns separate worker processes. The inter-process communication overhead is larger than the GPU compute time for our SDEs. Benchmarks showed single GPU is faster than 2-3 GPUs.

**Limitation:** L1DRAC system (with adaptive callback) remains CPU-only. GPU only works for Nominal and True system simulations.

**Usage:**
```julia
# Set at top of script before any GPU code runs
max_GPUs = 1

# Then run ensemble simulation
ens_nom_sol = system_simulation(params, nominal_system; simtype=:ensemble, backend=:gpu)
```

#### 2. Solver backend (`cpu()` `gpu()`)
 should this not be auto assigned since we have a code that determines GPU/CPU based on user input and available resources?

#### 3. `Warmup` for JIT compilaaion

#### 4. `@CUDA.time()` for GPU solvers, and at the end `GC.gc()` and `CUD.reclaim()` to free up GPU memory. 

#### 5. Solution returned are `Vector{<:RODESolution}` for a single trajectory solution, or `Vector{<:EnsembleSolution}` with `length = N`, `N = 1` when using CPU or single GPU, and `N = # of GPUs` used 




