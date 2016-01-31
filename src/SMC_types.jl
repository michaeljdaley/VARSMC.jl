type Likelihood
  sym::Symbol
  dependents::Array{Symbol,1}
  dist::Function
  data::Array
end

type Prior
  sym::Symbol
  dependents::Array{Symbol,1}
  dist::Distribution
  proposal_dist::Distribution
end

type Variable
  sym::Symbol
  dependents::Array{Symbol,1}
  eq::Function
end

Parameter = Union{Prior, Variable}

type Model
  particles::Dict{Symbol, Array}
  proposals::Dict{Symbol, Array}
  solve_order::Array{Symbol,1}
  parameters::Dict{Symbol,Parameter}
  likelihood::Likelihood
  num_particles::Integer
  num_iterations::Integer
  tempering_power::Float64
  tempering_schedule::Array{Float64,1}
  weights::Array{Float64,1}
end

# Constructor for Model type
# Assigns parameters, generates particles and sets tempering schedule
function Model(num_particles::Integer, num_iterations::Integer, tempering_power::Float64, like::Likelihood, args...)
  m = Model(
    Dict{Symbol, Array}(),
    Dict{Symbol, Array}(),
    Array{Symbol,1}(),
    Dict{Symbol,Parameter}(),
    like,
    num_particles,
    num_iterations,
    tempering_power,
    Array{Float64,1}(),
    Array{Float64,1}()
  )

  # Put variables into appropriate arrays
  for i in args
    m.parameters[i.sym] = i
  end

  # Generate tempering schedule
  m.tempering_schedule = generate_schedule(num_particles, num_iterations, tempering_power)

  return m
end
