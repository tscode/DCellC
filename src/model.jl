
# --------------------------------------------------------------------------- #
# Abstract type of a processing Model

abstract type Model{I <: Image} end

# --------------------------------------------------------------------------- #
# Get the image type that the model is constructed for

imgtype{I <: Image}(:: Model{I}) = I


# --------------------------------------------------------------------------- #
# Reference to the variables of a model trained by backpropagation

weights(m :: Model) = error("Not implemented")


# --------------------------------------------------------------------------- #
# Reference to other internals of a model

state(m :: Model) = error("Not implemented")


# --------------------------------------------------------------------------- #
# Forward evaluation of the model

# Low-level function needed to support automated differentiation
density(w, s, data, ::Type{Model}) = error("Not implemented")

# Convenience function built on top 
function density{I <: Image}(m :: Model{I}, img :: I)
  x = reshape(img.data, (imgsize(img)..., imgchannels(I), 1))
  return density(weights(m), state(m), x, typeof(m))[:,:,1]
end


# A full image may be too large to go through the model in a single run.
# So divide the image into patches, apply the model to the patches, and
# stich the density-patches together again.
function density_patched{I <: Image}(model :: Model{I}, img :: I; 
                                     patchsize = 256, overlap = 24,
                                     at = Array{Float32}, 
                                     callback :: Function = (i, n) -> nothing)

  s = (n, m) = imgsize(img)
  d = overlap

  # TODO: The first two can be relaxed
  @assert patchsize >= 48 "patchsize must be larger than or equal to 48"
  @assert min(n, m) > patchsize "image sizes must be larger than patchsize"
  @assert d >= 8 "overlap must be larger than or equal to 8"

  # Find parameters for suitable patch regions
  # k = number of patches in y/x direction
  # l = good length of the patches in y/x direction
  k = ceil.( Int, (s .- patchsize) ./ (patchsize .- d) ) .+ 1
  l = ceil.( Int, (s .+ (k .- 1) .* d) ./ (8 .* k) ) .* 8

  mirrorinds(a, l, n) = (a+l-1 <= n) ? (a:(a+l-1)) : ([a:n; n:-1:(n-(a+l-2-n))])

  # Pack the model, i.e. bring it to graphics memory if suitable at is given
  # TODO: These functions are only introduced in the file training.jl.
  # Should be moved?
  w = packweights(weights(model), at = at)
  s = packstate(state(model), at = at)

  # Create patch data
  patchdata = zeros(Float32, l..., imgchannels(img), prod(k))
  for i in 1:k[1], j in 1:k[2]
    y = (i == 1) ? 1 : (i-1)*(l[1] - d)
    x = (j == 1) ? 1 : (j-1)*(l[2] - d)

    # Get indices for this patch. If the patch would protrude from the
    # image, continue the patch on a mirror version after the image border
    yind = mirrorinds(y, l[1], n)
    xind = mirrorinds(x, l[2], m)

    patchdata[:,:,:,sub2ind((k[2], k[1]), j, i)] = imgdata(img)[yind,xind,:]
  end

  # Load the patch data to gpu if suitable at is given
  pd = convert(at, patchdata)

  # Bring the density patches in rectangular layout
  densities = Array{Array{Float32, 2}}(k...)
  for i in 1:k[1], j in 1:k[2]
    ind = sub2ind((k[2], k[1]), j, i)
    data = patchdata[:,:,:,ind:ind]
    densities[i, j] = convert(at, density(w, s, data, typeof(model)))[:,:,1]
    
    # Give some feedback about the status via callback
    callback(ind, prod(k))
  end

  # Merge the density patches with smooth interpolation profiles
  ftrans = [ 1 - sin(r*(pi/2) / (d-1))^2 for r in 1:d ]
  rtrans = reverse(ftrans)

  sprofile = [ones(patchsize - d); ]
  eprofile = sprofile[end:-1:1]
  mprofile = sprofile .* eprofile

  dens = zeros(Float32, (k .* l .- (k.-1) .* d)...)
  
  for i in 1:k[1], j in 1:k[2]

    # If we are at the first or last patches in the x or y direction,
    # use start or end profiles; use middle profiles otherwise
    if i == 1
      y = 1
      yprofile = [ones(l[1] - d); ftrans]
    elseif i == k[1]
      y = (i-1)*(l[1] - d)
      yprofile = [rtrans; ones(l[1] - d)]
    else
      y = (i-1)*(l[1] - d)
      yprofile = [rtrans; ones(l[1] - 2d); ftrans] 
    end

    if j == 1
      x = 1
      xprofile = [ones(l[2] - d); ftrans]
    elseif j == k[2]
      x = (j-1)*(l[2] - d)
      xprofile = [rtrans; ones(l[2] - d)]
    else
      x = (j-1)*(l[2] - d)
      xprofile = [rtrans; ones(l[2] - 2d); ftrans] 
    end

    profile = yprofile .* xprofile'
    dens[y:y+l[1]-1, x:x+l[2]-1] += profile .* densities[i, j]

  end

  return dens[1:n, 1:m]
end


# --------------------------------------------------------------------------- #
# Estimate labels
# Densities are the output of the application of a neural network regressor

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
  dmap = density(args...; kwargs...)
  return label(convert(Array{Float32}, dmap))
end


# --------------------------------------------------------------------------- #
# Some auxiliary functions for building neural network models

include("models/primitives.jl")


# --------------------------------------------------------------------------- #
# Class of models from (Pen, Yang, Li, et al.; 2018) 
# and (Xie, Noble, Zisserman; 2015)

abstract type FCModel{I, BN} <: Model{I} end

weights(m :: FCModel) = m.weights
state(m :: FCModel) = m.bnmoments
batchnorm{I, BN}(:: FCModel{I, BN}) = BN


# Different architectures of this model class

include("models/unetlike.jl")
include("models/multiscale3.jl")
include("models/fcrna.jl")

