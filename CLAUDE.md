# L1DRAC.jl - Project Configuration

## Project Overview
Julia package implementing L1 Distributionally Robust Adaptive Control (L1-DRAC) algorithms from the paper by Gahlawat, Karumanchi, and Hovakimyan (2025).

## Related Resources

### Manuscript (READ-ONLY)
- **Location:** `~/Desktop/GitRepos/DRAC_WriteUps/ManuscriptFinal.tex`
- **Permission:** Read only - DO NOT modify this file
- Use this manuscript as the theoretical reference for implementation details

## Current Development Focus
See `TODO.md` for active tasks:
- Nonlinear optimization problem for minimum rho_r, rho_a, omega
- GitHub Pages setup
- Sample path plotting with covariance ellipsoids
- 2-Wasserstein distance computation and bound verification

## Project Structure
```
L1DRAC/
├── src/
│   ├── L1DRAC.jl      # Main module
│   ├── types.jl       # Type definitions
│   ├── simfunctions.jl # Simulation functions
│   └── L1functions.jl  # L1 adaptive control functions
├── test/              # Test suite
├── examples/          # Example implementations (IGNORE - not part of package)
└── TODO.md            # Active task list
```

## Important Notes
- **IGNORE `examples/AFOSR_2025/`** - This directory is NOT part of the package, do not reference it for implementation patterns

## Julia Conventions
- Follow standard Julia style guide
- Use `@unpack` macro from UnPack.jl for struct destructuring
- Leverage DifferentialEquations.jl for SDE/ODE solving
- Export public API functions from main module

## Testing
Run tests with:
```julia
using Pkg
Pkg.test("L1DRAC")
```

## Collaborators
- Aditya Gahlawat
- Sambhu
