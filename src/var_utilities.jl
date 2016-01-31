# Create observation and data matrix in standard VAR form
# from a given set of observations and lags
function data_matrix(observations, lags)
  # First column for constants
  data = ones(size(observations,1)-lags,1)

  for i in 1:lags
    data = hcat(data, observations[(lags-i+1):(end-i),:])
  end

  observations = observations[(lags+1):end,:]
  return observations, data
end

# Create the mean and variance arrays for a Minnesota prior.
# See "Forecasting and Conditional Projection Using
# Realistic Prior Distributions" by Doan, Litterman and Sims (1984)
function minnesota_prior(Y, X, λ1, λ2, λ3, λ4)
  numvars = size(Y,2)
  params = size(X,2)
  numlags = div((params - 1),numvars)
  samplesize = size(Y,1)

  se = zeros(numvars)

  for i in 1:numvars
    ys = Y[:,i]
    xs = hcat(X[:,1], X[:,i+1])
    βs = (xs'*xs)\(xs'*ys)
    ϵs = (ys-(xs*βs))'*(ys-(xs*βs))
    se[i] = sqrt(ϵs[1]/(samplesize-2))
  end

  # Set up prior means for own lags
  βp = zeros(params, numvars)
  for i in 1:numvars
    βp[i + 1, i] = 1
  end

  βp = vec(βp)

  Σp = zeros(numvars*params, numvars*params)

  for i in 1:numvars
    loc = 1 + (i - 1)*params
    Σp[loc, loc] = (λ4*se[i])^2

    for k in 1:numlags
      for j in 1:numvars
        loc = j + 1 + (k - 1)*numlags + (i - 1)*params

        if (i == j)
          Σp[loc, loc] = (λ1/(k^λ3))^2
        else
          Σp[loc, loc] = (se[i]*λ1*λ2/(se[j]*k^λ3))^2
        end
      end
    end
  end

  return βp, Σp
end
