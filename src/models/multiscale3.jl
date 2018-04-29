
# --------------------------------------------------------------------------- #
# The "Multiscale_3layers" model from (Pan, Yang, Li, et al.; 2018)

struct Multiscale3{I, BN} <: FCModel{I, BN}
  weights :: Vector{Any}
  bnmoments :: Vector{Any}
end


# --------------------------------------------------------------------------- #
# Network architecture (no batch normalization)

function density{I <: Image}(w, s, x, ::Type{Multiscale3{I, false}})
  y1 = rconv(w[2], rconv(w[1], x))
  y2 = rconv(w[4], pool(rconv(w[3], x)))
  y3 = rconv(w[6], rconv(w[5], pool(y1)))

  y4 = rconv(w[8], rconv(w[7], cat3(pool(y2), pool(y3))))
  y5 = rconv(w[11], rconv(w[10], uconv(w[9], y3, y4)))
  y6 = rconv(w[14], rconv(w[13], uconv(w[12], y1, y5)))

  y = conv4(w[15], y6, padding=pad(w[15]))
  return reshape(y, size(y)[[1,2,4]]...)
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
    wr(c, 64), wr(64, 64), wr(c, 64, (11,11)), wr(64, 64, (9,9)),
    wr(64, 64), wr(64, 64), wr(128, 64), wr(64, 64), wu(64, 64), 
    wr(128, 64), wr(64, 64), wu(64, 64), wr(128, 64), wr(64, 64),
    wr(64, 1) ]
end

init_state{I<:Image}(::Type{Multiscale3{I, false}}) = Any[]


# --------------------------------------------------------------------------- #
# Network architecture (batch normalization)

function density{I <: Image}(w, s, x, ::Type{Multiscale3{I, true}})
  y1 = rbconv(w[4], s[2], w[3], rbconv(w[2], s[1], w[1], x))
  y2 = rbconv(w[8], s[4], w[7], pool(rbconv(w[6], s[3], w[5], x)))
  y3 = rbconv(w[12], s[6], w[11], rbconv(w[10], s[5], w[9], pool(y1)))

  u4 = cat3(pool(y2), pool(y3))
  y4 = rbconv(w[16], s[8], w[15], rbconv(w[14], s[7], w[13], u4))

  u5 = ubconv(w[18], s[9], w[17], y3, y4)
  y5 = rbconv(w[22], s[11], w[21], rbconv(w[20], s[10], w[19], u5))

  u6 = ubconv(w[24], s[12], w[23], y1, y5)
  y6 = rbconv(w[28], s[14], w[27], rbconv(w[26], s[13], w[25], u6))

  y = conv4(w[29], y6, padding=pad(w[29]))
  return reshape(y, size(y)[[1,2,4]]...)
end


# --------------------------------------------------------------------------- #
# Initialization (batch normalization)

function init_weights{I<:Image}(::Type{Multiscale3{I, true}}, 
                                wr :: Function = wr, 
                                wu :: Function = wu; 
                                seed :: Integer = rand(1:10000))

  # Need the same convolutional weights as in the case without batch
  # normalization. However, also need to insert the trainable bn-values at
  # the right positions of the weights vector

  weights = init_weights(Multiscale3{I, false}, wr, wu, seed = seed)
  return Any[ odd(i) ? weights[ceil(Int, i/2)] : bn(64) for i in 1:29 ]
end

function init_state{I<:Image}(::Type{Multiscale3{I, true}}) 
  return Any[ bnmoments() for i in 1:14 ]
end


# --------------------------------------------------------------------------- #
# Convenience-constructor

function Multiscale3(I,
                     wr :: Function = wr, 
                     wu :: Function = wu; 
                     bn :: Bool = false,
                     seed :: Integer = rand(1:10000))

  weights = init_weights(Multiscale3{I, bn}, wr, wu, seed = seed)
  state = init_state(Multiscale3{I, bn})

  return Multiscale3{I, bn}(weights, state)
end

