function eval_parameter!(prior::Prior, model::Model, prior_pdfs::Array, particle::Integer)
  push!(prior_pdfs, logpdf(prior.dist, model.particles[prior.sym][particle]))
end

function eval_parameter(likelihood::Likelihood, model::Model, particle::Integer)
  return loglikelihood(likelihood.dist([model.particles[sym][particle] for sym in likelihood.dependents]...), likelihood.data)
end

function update_weights!(model::Model, weights::Array, iteration::Integer)
  num_particles = model.num_particles
  tempering_coefficient = model.tempering_schedule[iteration] - model.tempering_schedule[iteration-1]

  # Change this from a dict...probably just needs to be an array
  # Can I fix the size of this?

  for i in 1:num_particles
    # Change this from a dict...probably just needs to be an array
    # Can I fix the size of this?
    prior_pdfs = Float64[]

    for sym in model.solve_order
      if typeof(model.parameters[sym]) == Prior
        eval_parameter!(model.parameters[sym], model, prior_pdfs, i)
      end
    end

    loglike = eval_parameter(model.likelihood, model, i)

    weights[i] = exp(tempering_coefficient*loglike + sum(prior_pdfs))
  end
end

function resample!(weights::Array, model::Model)
  num_particles = model.num_particles

  # Resample via multinomial resampling
  resample = rand(Categorical(weights/sum(weights)),num_particles)

  # Update particles for each parameter
  for (key, value) in model.particles
    new_particles = similar(value)

    for (i, j) in enumerate(resample)
      new_particles[i] = value[j]
    end

    model.particles[key] = new_particles
  end
end

function mh_propagation!(model::Model, steps::Integer)
  #println("Into MH")
  mh_acceptances = 0
  mh_errors = 0
  priors = filter((k, v) -> typeof(v) == Prior, model.parameters)

  for j = 1:steps
    for i in 1:model.num_particles
      proposal_pdfs = Float64[]
      previous_pdfs = Float64[]

      try
      # Move particles and evaluate priors
        #println("creating proposals")
        for var in model.solve_order
          if typeof(model.parameters[var]) == Prior
            model.proposals[var][i] = model.particles[var][i] + reshape(rand(model.parameters[var].proposal_dist),size(model.particles[var][i]))
            push!(proposal_pdfs, logpdf(model.parameters[var].dist, model.proposals[var][i]))
            push!(previous_pdfs, logpdf(model.parameters[var].dist, model.particles[var][i]))
          elseif typeof(model.parameters[var]) == Variable
            model.proposals[var][i] = model.parameters[var].eq([model.proposals[sym][i] for sym in model.parameters[var].dependents]...)
          end
        end

        #println("got to try loop")
        push!(previous_pdfs, loglikelihood(model.likelihood.dist([model.particles[sym][i] for sym in model.likelihood.dependents]...), model.likelihood.data))
        #println("got to proposal loop")
        push!(proposal_pdfs, loglikelihood(model.likelihood.dist([model.proposals[sym][i] for sym in model.likelihood.dependents]...), model.likelihood.data))

        α = min(exp(sum(proposal_pdfs) - sum(previous_pdfs)),1)
        #println(string("α: ", α))

        if α > rand(Uniform(0,1))
          mh_acceptances = mh_acceptances + 1
          for var in model.solve_order
            model.particles[var][i] = model.proposals[var][i]
          end
        end
      catch err2
        # On error, don't accept the particle
        mh_errors = mh_errors + 1
        #println(string("Warning: ", err2))
        #println("Check MH proposal distribution for covariance matrices")
      end
    end
  end

  println(string("Acceptances: ", mh_acceptances))
  println(string("Errors: ", mh_errors))
  #println("Out of MH")
end

function run_smc(model::Model)
  num_iterations = model.num_iterations
  num_particles = model.num_particles

  priors = filter((k, v) -> typeof(v) == Prior, model.parameters)

  variables = filter((k, v) -> typeof(v) == Variable, model.parameters)

  weights = ones(num_particles)
  new_weights = zeros(num_particles)



  # This needs to change...
  # Set up arrays for resampling. Probably put it in
  # The model type
  new_particles = Dict()

  # Begin main iterative loop
  for n = 2:num_iterations
    # Calculate new weights
    try
      update_weights!(model, new_weights, n)
    catch err3
      println("weights update failed")
      throw(err3)
    end

    # Normalise weights
    #=
    try
      denom = dot(new_weights, weights)/num_particles
    catch err4
      println("denom failed")
      throw(err4)
    end
    =#

    try
      denom = dot(new_weights, weights)/num_particles
      for i in 1:num_particles
        weights[i] = new_weights[i]*weights[i]/denom
      end
    catch err5
      println("weight normalisation failed")
      throw(err5)
    end

    model.weights = weights

    # Calculate effective sample size
    ESS = num_particles/((sum(weights.^2))/num_particles)

    println(ESS)

    # Determine if the particles need to be resampled
    if ESS < (num_particles/2) # Change this to be set by user
      try
        resample!(weights, model)
      catch err6
        println("resampling failed")
        throw(err6)
      end
      weights = ones(num_particles)
    end

    # Propagate particles
    try
      mh_propagation!(model, 1)
    catch err7
      println("mh propagation failed")
      throw(err7)
    end
  end
end
