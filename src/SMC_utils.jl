function generate_schedule(num_particles, num_iterations, tempering_power)
  tempering_schedule = zeros(num_iterations)

  for i in 1:num_iterations
      tempering_schedule[i] = ((i-1)/(num_iterations-1))^tempering_power
  end

  return tempering_schedule
end


function generate_particles(prior::Prior, model::Model)
  particles = Array[]

  for i in 1:model.num_particles
    push!(particles, rand(prior.dist))
  end

  return particles
end

function generate_particles(var::Variable, model::Model)
  particles = Array[]

  for i in 1:model.num_particles
    push!(particles, var.eq([model.particles[sym][i] for sym in var.dependents]...))
  end

  return particles
end

function set_solve_order!(model::Model, solve_order::Array)
  # Remove likelihood from the solve order if it has been included
  model.solve_order = filter(sym -> sym != model.likelihood.sym, solve_order)

  # Generate initial set of particles
  for sym in model.solve_order
    particles = generate_particles(model.parameters[sym], model)
    model.particles[sym] = particles
    model.proposals[sym] = particles
  end
end
