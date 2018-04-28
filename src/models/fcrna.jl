
# --------------------------------------------------------------------------- #
# Inspired by model "FCRN-A" model from (Xie, Noble, Zisserman; 2015)

struct FCRNA{I <: Image, BN} <: FCModel{I, BN}
  weights :: Vector{Any}
  bnmoments :: Vector{Any}
end


# --------------------------------------------------------------------------- #
# Network architecture (no batch normalization)

function densitymap{I <: Image}(w, s, x, ::Type{FCRNA{I, false}})
  y = rconv(w[2],  pool(rconv(w[1], x)))
  y = rconv(w[4],  pool(rconv(w[3], pool(y))))
  y = rconv(w[6],  uconv(w[5], y))
  y = rconv(w[8],  uconv(w[7], y))
  y = rconv(w[10], uconv(w[9], y))
  
  return conv4(w[11], y, padding=pad(w[11]))[:,:,1,:]
end


# --------------------------------------------------------------------------- #
# Initialization (no batch normalization)

function init_weights{I<:Image}(::Type{FCRNA{I, false}}, 
                                wr :: Function = wr, 
                                wu :: Function = wu; 
                                seed :: Integer = rand(1:10000))
  Knet.setseed(seed)
  c = imgchannels(I)

  return Any[
    wr(c, 32), wr(32, 64), wr(64, 128), wr(128, 512), 
    wu(512, 512), wr(512, 128), wu(128, 128), wr(128, 64), 
    wu(64, 64), wr(64, 32), wr(32, 1) ]
end

init_state{I<:Image}(::Type{FCRNA{I, false}}) = Any[]


# --------------------------------------------------------------------------- #
# Network architecture (batch normalization)

function densitymap{I <: Image}(w, s, x, ::Type{FCRNA{I, true}})
  y = rbconv(w[4],  s[2],  w[3],  pool(rbconv(w[2], s[1], w[1], x)))
  y = rbconv(w[8],  s[4],  w[7],  pool(rbconv(w[6], s[3], w[5], pool(y))))
  y = rbconv(w[12], s[6],  w[11], ubconv(w[10], s[5], w[9], y))
  y = rbconv(w[16], s[8],  w[15], ubconv(w[14], s[7], w[13], y))
  y = rbconv(w[20], s[10], w[19], ubconv(w[18], s[9], w[17], y))
  
  return conv4(w[21], y, padding=pad(w[21]))[:,:,1,:]
end


# --------------------------------------------------------------------------- #
# Initialization (batch normalization)

function init_weights{I<:Image}(::Type{FCRNA{I, true}}, 
                                wr :: Function = wr, 
                                wu :: Function = wu; 
                                seed :: Integer = rand(1:10000))

  # Need the same convolutional weights as in the case without batch
  # normalization. However, also need to insert the trainable bn-values at
  # the right positions of the weights vector

  bm = [32, 64, 128, 512, 512, 128, 128, 64, 64, 32]
  weights = init_weights(FCRNA{I, false}, wr, wu, seed = seed)
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
               seed :: Integer = rand(1:10000))

  weights = init_weights(FCRNA{I, bn}, wr, wu, seed = seed)
  state = init_state(FCRNA{I, bn})

  return FCRNA{I, bn}(weights, state)
end

