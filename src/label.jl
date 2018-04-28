
# --------------------------------------------------------------------------- #
# Model interface functions 

struct Label
  data :: Matrix{Int}
  function Label(data)
    @assert (size(data, 1) == 2)
    new(data)
  end
end

Label() = Label(zeros(Int, 2, 0))


# --------------------------------------------------------------------------- #
# Estimate labels 

function label(dens :: Array{Float32, 2}; 
               candidates = false, level = 25, clevel = 10) 

  # Find local maxima that are sufficiently high

  label = Array{Int}[]
  cands = Array{Int}[]

  m, n = size(dens)

  for i in 1:m, j in 1:n
    xsel = max(i-1, 1):min(i+1, m)
    ysel = max(j-1, 1):min(j+1, n)

    if dens[i, j] < maximum(dens[xsel, ysel])
      continue
    end

    if dens[i, j] >= level
      push!(label, [j, i])
    elseif dens[i, j] >= clevel
      push!(cands, [j, i])
    end
  end

  label = isempty(label) ? zeros(Int, 2, 0) : hcat(label...)
  cands = isempty(cands) ? zeros(Int, 2, 0) : hcat(cands...)

  if candidates
    return Label(label), Label(cands)
  else
    return Label(label)
  end
end

function label(dens :: Array{Float32, 3}; kwargs...)
  return label.([dens[:,:,i] for i in 1:size(dens, 3)]; kwargs...)
end

function label(args...; kwargs...)
  dmap = densitymap(args...; kwargs...)
  return label(convert(Array{Float32}, dmap))
end


# --------------------------------------------------------------------------- #
# Performance of density-based label compared to true label
# "mean smallest distance"

function adjacency(dlbl, tlbl)
  if isempty(dlbl.data) && isempty(tlbl.data)
    return 0.
  elseif isempty(dlbl.data) || isempty(tlbl.data)
    return Inf
  end

  n, m = length(tlbl), length(dlbl)
  vals = zeros(n + m)
  for i in 1:n
    dists = sqrt.(sum(abs2, dlbl.data .- tlbl.data[:,i], 1))[1,:]
    vals[i] = minimum(dists)
  end
  for i in 1:m
    dists = sqrt.(sum(abs2, dlbl.data[:,i] .- tlbl.data, 1))[1,:]
    vals[n+i] = minimum(dists)
  end
  return vals
end

function adjacency(model, img :: Image, lbl :: Label; kwargs...)
  return adjacency(label(model, img; kwargs...), lbl)
end


# --------------------------------------------------------------------------- #
# Convenience functions

Base.size(l :: Label, args...; kwargs...) = size(l.data, args..., kwargs...)
Base.length(l :: Label, args...; kwargs...) = div(length(l.data, args..., kwargs...), 2)
Base.getindex(l :: Label, k :: Integer) = l.data[:,k]

