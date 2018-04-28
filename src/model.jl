
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
densitymap(w, s, data, ::Type{Model}) = error("Not implemented")

# Convenience function built on top 
function densitymap{I <: Image}(m :: Model{I}, img :: I)
  x = reshape(img.data, (imgsize(img)..., imgchannels(I), 1))
  return densitymap(weights(m), state(m), x, typeof(m))[:,:,1]
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

