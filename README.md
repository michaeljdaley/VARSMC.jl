# VARSMC.jl
Bayesian VARs via Sequential Monte Carlo

Uses an SMC algorithm similar to that in "Bayesian Estimation of DSGE Models" by Herbst & Schorfheide to estimate vector autoregressions.

## Example

Need to use Distributions as well for specifying priors and likelihoods.

```julia
using VARSMC
using Distributions
```

There are some convenience functions for vector autoregressions. data_matrix(data, lags) creates standard Y, X matrices for a VAR.
minnesota_prior(Y, X, lamba1, lambda2, lambda3, lambda4) creates priors mean and covariance matrics for B based on the prior of Doan, Litterman and Sims.

```julia
data = readcsv("datain.csv")

Y, X = data_matrix(data, 2)
βp, Σp = minnesota_prior(Y, X, 1, 1, 1, 1)
```

Create a likelihood using the Likelihood type. Requires a symbol, symbols of parameters the likelihood depends on, a function for the distribution and the data for the likelihood.

```julia
y = Likelihood(:Y, [:μ, :H], (μ, H) -> MvNormal(μ, H), reshape(Y, 500, 1))
```

Create priors using the Prior type. Requires a symbol, dependencies if any, a distribution for the prior and a distribution for the Metropolis Hastings steps.

```julia
b = Prior(:β, [], MvNormal(βp, Σp), MvNormal(zeros(10), 0.2*eye(10)))
s = Prior(:Σ, [], InverseWishart(3, eye(2)), MvNormal(zeros(4), 0.01*eye(4)))
```

Create any other parameters using the Variable type. For example, here we create the means and covariance matrices that feed into the likelihood.

```julia
Ib = eye(2)
Is = eye(250)

h = Variable(:H, [:Σ], (Σ) -> kron(Is,Σ))
mu = Variable(:μ, [:β], (β) -> kron(Ib,X)*β)
```

Gather all the parameters and create a model. Requires the number of particles, number of iterations, tempering parameter and then the parameters for the likelihood, prior etc.

```julia
m = Model(
  1000,
  100,
  2.0,
y, b, s, h, mu
)
```

Set the solve order so that the SMC algorithm solves the different equations in the correct order, and then run the SMC algorithm using run_smc.

```julia
set_solve_order!(m, [:β, :Σ, :μ, :H, :Y])

run_smc(m)
```

The weights and particles can be accessed via:

```julia
m.weights
m.particles
```
