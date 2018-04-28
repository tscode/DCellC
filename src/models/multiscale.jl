
abstract type Multiscale{I, BN} <: Model{I} end

# --------------------------------------------------------------------------- #
# The "Multiscale_3layers" model from (Pen, Yang, Li, et al.; 2018)

struct Multiscale3{I, BN} <: Multiscale{I, BN}
  weights :: Array
  bnmoments :: Array
end


# --------------------------------------------------------------------------- #
# Access model internals

weights(m :: UNetLike) = m.weights
state(m :: UNetLike) = m.bnmoments


# --------------------------------------------------------------------------- #
# General interface for the density map
# Calls the specific implementations provided below

function densitymap{I <: Image}(m :: Multiscale3{I}, img :: I)
  x = reshape(img.data, (size(img.data)[1:2]..., imgchannels(I), 1))
  x = convert(Array{Float32}, x)
  return densitymap(weights(m), state(m), x, typeof(m))[:,:,1]
end


# --------------------------------------------------------------------------- #
# Network architecture (no batch normalization)

function densitymap{I <: Image}(w, s, x, ::Type{Multiscale3{I, false}})
  y1 = rconv(w[2], rconv(w[1], x))
  y2 = rconv(w[4], pool(rconv(w[3], x, size=(11,11))), size=(9,9))
  y3 = rconv(w[6], rconv(w[5], y1))

  y4 = rconv(w[8], rconv(w[7], cat(3, pool(y2), pool(y3))))
  y5 = rconv(w[11], rconv(w[10], uconv(w[9], y3, y4)))
  y6 = rconv(w[14], rconv(w[13], uconv(w[12], y1, y5)))

  return conv4(w[15], y6, padding=pad(w[15]))[:,:,1,:]
end

# --------------------------------------------------------------------------- #
# Initialization (no batch normalization)

function init_weights{I<:Image}(::Type{Multiscale3{I, false}}, 
                                wr :: Function = wr, 
                                wu :: Function = wu; 
                                seed :: Integer = rand(1:10000))
  Knet.setseed(seed)
  c = imgchannels(I)

  return Any[
    
end

init_state{I<:Image}(::Type{UNetLike{I, false}}) = Any[]

