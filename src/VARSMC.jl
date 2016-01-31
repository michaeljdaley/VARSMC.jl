module VARSMC

using Distributions

include("SMC_types.jl")
include("SMC_utils.jl")
include("SMC_algorithm.jl")

export
  Likelihood,
  Prior,
  Parameter,
  Variable,
  Model,
  set_solve_order!
  run_smc

# Module end
end
