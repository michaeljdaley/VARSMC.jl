module VARSMC

using Distributions

include("SMC_types.jl")
include("SMC_utils.jl")
include("SMC_algorithm.jl")
include("var_utilities.jl")

export
  Likelihood,
  Prior,
  Parameter,
  Variable,
  Model,
  set_solve_order!,
  run_smc,
  data_matrix,
  minnesota_prior

# Module end
end
