
# --------------------------------------------------------------------------- #
# Abstract type of a processing Model

abstract type Model{I <: Image} end


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


# --------------------------------------------------------------------------- #
# Saving and loading

function save(model :: Model, fname :: String; description :: String = "")
  Jld2.jldopen(fname, true, true, true, IOStream) do file
    write(file, "model", model)
    write(file, "description", description)
  end
end

function load(fname :: String; description :: Bool = false)
  d = description
  JLD2.@load fname model description
  if d return model, description
  else return model
  end
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
# Auxilliary functions to implement fully convolutional neural networks

# Padding for convolution such that dimensions are preserved
pad(w) = floor(Int, size(w, 1) / 2)

# Convolution and relu activation
rconv(w, x) = relu.(conv4(w, x, padding=pad(w))) 

# Convolution, batch normalization, and relu activation
rbconv(wb, m, wc, x) = relu.(batchnorm(conv4(wc, x, padding=pad(wc)), m, wb))

# Up-convolution
uconv(w, x) = deconv4(w, x, stride=2, padding=1)

# Up-convolution and batch normalization
ubconv(wb, m, wc, x) = batchnorm(deconv4(wc, x, stride=2, padding=1), m, wb)

# Up-convolution and stacking
uconv(w, x, y) = cat(3, x, deconv4(w, y, stride=2, padding=1))

# Up-convolution, batch normalization, and stacking
ubconv(wb, m, wc, x, y) = 
    cat(3, x, batchnorm(deconv4(wc, y, stride=2, padding=1), m, wb))


# Default initialization for convolution kernel
wr(i, o, size = (3, 3)) = gaussian(Float32, size..., i, o, std=0.060)

# Default initialization for up-convolution kernel
wu(i, o) = bilinear(Float32, 2, 2, o, i)

# Knet parameters for batch normalization
bn(c) = bnparams(Float32, c)


# Check if a number is odd 
odd(x::Integer) = (x % 2 != 0)


# --------------------------------------------------------------------------- #
# Class of models from (Pen, Yang, Li, et al.; 2018) 
# and (Xie, Noble, Zisserman; 2015)

abstract type FCModel{I, BN} <: Model{I} end

weights(m :: FCModel) = m.weights
state(m :: FCModel) = m.bnmoments

# Different architectures of this model class
include("models/unetlike.jl")
include("models/multiscale3.jl")
include("models/fcrna.jl")

