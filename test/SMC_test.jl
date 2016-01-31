using VARSMC
using Distributions

# Get data
data = readcsv("datain.csv")

Y, X = data_matrix(data, 2)
βp, Σp = minnesota_prior(Y, X, 1, 1, 1, 1)

# Create parameters
y = Likelihood(:Y, [:μ, :H], (μ, H) -> MvNormal(μ, H), reshape(Y, 500, 1))
b = Prior(:β, [], MvNormal(βp, Σp), MvNormal(zeros(10), 0.2*eye(10)))
s = Prior(:Σ, [], InverseWishart(3, eye(2)), MvNormal(zeros(4), 0.01*eye(4)))

Ib = eye(2)
Is = eye(250)

h = Variable(:H, [:Σ], (Σ) -> kron(Is,Σ))
mu = Variable(:μ, [:β], (β) -> kron(Ib,X)*β)

# Create model
m = Model(
  1000,
  100,
  2.0,
y, b, s, h, mu
)

# Set solve order (e.g. so don't try and calculate μ before β)
set_solve_order!(m, [:β, :Σ, :μ, :H, :Y])

# Estimate
run_smc(m)
