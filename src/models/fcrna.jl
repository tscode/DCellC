
# --------------------------------------------------------------------------- #
# Inspired by model "FCRN-A" model from (Xie, Noble, Zisserman; 2015)

struct FCRNA{I <: Image, BN} <: FCModel{I, BN}
  weights :: Vector{Any}
  bnmoments :: Vector{Any}
end


# --------------------------------------------------------------------------- #
# Network architecture (no batch normalization)

function density{I <: Image}(w, s, x, ::Type{FCRNA{I, false}})
  y = rconv(w[2],  pool(rconv(w[1], x)))
  y = rconv(w[4],  pool(rconv(w[3], pool(y))))
  y = rconv(w[6],  uconv(w[5], y))
  y = rconv(w[8],  uconv(w[7], y))
  y = rconv(w[10], uconv(w[9], y))
  
  y = conv4(w[11], y, padding=pad(w[11]))
  return reshape(y, size(y)[[1,2,4]]...)
end


# --------------------------------------------------------------------------- #
# Initialization (no batch normalization)

function init_weights{I<:Image}(::Type{FCRNA{I, false}}, 
                                wr :: Function = wr, 
                                wu :: Function = wu; 
                                potency :: Integer = 6,
                                seed :: Integer = rand(1:10000))
  Knet.setseed(seed)
  c = imgchannels(I)
  s = 2^(potency-1)

  return Any[
    wr(c, s), wr(s, 2s), wr(2s, 4s), wr(4s, 16s), 
    wu(16s, 16s), wr(16s, 4s), wu(4s, 4s), wr(4s, 2s), 
    wu(2s, 2s), wr(2s, s), wr(s, 1) ]
end

init_state{I<:Image}(::Type{FCRNA{I, false}}) = Any[]


# --------------------------------------------------------------------------- #
# Network architecture (batch normalization)

function density{I <: Image}(w, s, x, ::Type{FCRNA{I, true}})
  y = rbconv(w[4],  s[2],  w[3],  pool(rbconv(w[2], s[1], w[1], x)))
  y = rbconv(w[8],  s[4],  w[7],  pool(rbconv(w[6], s[3], w[5], pool(y))))
  y = rbconv(w[12], s[6],  w[11], ubconv(w[10], s[5], w[9], y))
  y = rbconv(w[16], s[8],  w[15], ubconv(w[14], s[7], w[13], y))
  y = rbconv(w[20], s[10], w[19], ubconv(w[18], s[9], w[17], y))
  
  y = conv4(w[21], y, padding=pad(w[21]))
  return reshape(y, size(y)[[1,2,4]]...)
end


# --------------------------------------------------------------------------- #
# Initialization (batch normalization)

function init_weights{I<:Image}(::Type{FCRNA{I, true}}, 
                                wr :: Function = wr, 
                                wu :: Function = wu; 
                                potency :: Integer = 6,
                                seed :: Integer = rand(1:10000))

  # Need the same convolutional weights as in the case without batch
  # normalization. However, also need to insert the trainable bn-values at
  # the right positions of the weights vector
  s = 2^(potency-1)

  bm = [s, 2s, 4s, 16s, 16s, 4s, 4s, 2s, 2s, s]
  weights = init_weights(FCRNA{I, false}, wr, wu, 
                         potency = potency, seed = seed)
  return Any[ odd(i) ? weights[ceil(Int, i/2)] : bn(bm[div(i,2)]) for i in 1:21 ]
end

function init_state{I<:Image}(::Type{FCRNA{I, true}}) 
  return Any[ bnmoments() for i in 1:10 ]
end


# --------------------------------------------------------------------------- #
# Convenience-constructor

function FCRNA(I,
               wr :: Function = wr, 
               wu :: Function = wu; 
               bn :: Bool = false,
               potency :: Integer = 6,
               seed :: Integer = rand(1:10000))

  weights = init_weights(FCRNA{I, bn}, wr, wu, 
                         potency = potency, seed = seed)
  state = init_state(FCRNA{I, bn})

  return FCRNA{I, bn}(weights, state)
end

