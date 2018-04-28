
# --------------------------------------------------------------------------- #
# The "U-Net-like" model from (Pen, Yang, Li, et al.; 2018)

struct UNetLike{I, BN} <: FCModel{I, BN}
  weights :: Vector{Any}     # weights of the convolutional layers
  bnmoments :: Vector{Any}   # knet moment objects used for batch normalization
end


# --------------------------------------------------------------------------- #
# Network architecture (no batch normalization)

function density{I <: Image}(w, s, x, ::Type{UNetLike{I, false}})
  y1 = rconv(w[2], rconv(w[1], x))
  y2 = rconv(w[4], rconv(w[3], pool(y1)))
  y3 = rconv(w[6], rconv(w[5], pool(y2)))
  y4 = rconv(w[8], rconv(w[7], pool(y3)))

  z3 = rconv(w[11], rconv(w[10], uconv(w[9], y3, y4)))
  z2 = rconv(w[14], rconv(w[13], uconv(w[12], y2, z3))) 
  z1 = rconv(w[17], rconv(w[16], uconv(w[15], y1, z2)))

  return conv4(w[18], z1, padding=pad(w[18]))[:,:,1,:]
end


# --------------------------------------------------------------------------- #
# Initialization (no batch normalization)

function init_weights{I <: Image}(::Type{UNetLike{I, false}}, 
                                  wr :: Function = wr, 
                                  wu :: Function = wu; 
                                  seed :: Integer = rand(1:10000))

  Knet.setseed(seed)
  c = imgchannels(I)

  return Any[
    wr(c, 64),   wr(64, 64),  wr(64, 64), wr(64, 64), 
    wr(64, 64),  wr(64, 64),  wr(64, 64), wr(64, 64),
    wu(64, 64),  wr(128, 64), wr(64, 64), wu(64, 64),
    wr(128, 64), wr(64, 64),  wu(64, 64), wr(128, 64),
    wr(64, 64),  wr(64, 1) ]
end

init_state{I<:Image}(::Type{UNetLike{I, false}}) = Any[]


# --------------------------------------------------------------------------- #
# Network architecture (batch normalization)

function density{I <: Image}(w, s, x, ::Type{UNetLike{I, true}})
  y1 = rbconv(w[4],  s[2], w[3],  rbconv(w[2],  s[1], w[1],  x))
  y2 = rbconv(w[8],  s[4], w[7],  rbconv(w[6],  s[3], w[5],  pool(y1)))
  y3 = rbconv(w[12], s[6], w[11], rbconv(w[10], s[5], w[9],  pool(y2)))
  y4 = rbconv(w[16], s[8], w[15], rbconv(w[14], s[7], w[13], pool(y3)))

  u3 = ubconv(w[18], s[9], w[17], y3, y4)
  z3 = rbconv(w[22], s[11], w[21], rbconv(w[20], s[10], w[19], u3))

  u2 = ubconv(w[24], s[12], w[23], y2, z3)
  z2 = rbconv(w[28], s[14], w[27], rbconv(w[26], s[13], w[25], u2))

  u1 = ubconv(w[30], s[15], w[29], y1, z2)
  z1 = rbconv(w[34], s[17], w[33], rbconv(w[32], s[16], w[31], u1))

  return conv4(w[35], z1, padding=pad(w[35]))[:,:,1,:]
end


# --------------------------------------------------------------------------- #
# Initialization (batch normalization)

function init_weights{I<:Image}(::Type{UNetLike{I, true}}, 
                                wr :: Function = wr, 
                                wu :: Function = wu; 
                                seed :: Integer = rand(1:10000))

  # Need the same convolutional weights as in the case without batch
  # normalization. However, also need to insert the trainable bn-values at
  # the right positions of the weights vector

  weights = init_weights(UNetLike{I, false}, wr, wu, seed = seed)
  return Any[ odd(i) ? weights[ceil(Int, i/2)] : bn(64) for i in 1:35 ]
end

function init_state{I<:Image}(::Type{UNetLike{I, true}}) 
  return Any[ bnmoments() for i in 1:17 ]
end



# --------------------------------------------------------------------------- #
# Convenience-constructor

function UNetLike(I,
                  wr :: Function = wr, 
                  wu :: Function = wu; 
                  bn :: Bool = false,
                  seed :: Integer = rand(1:10000))

  weights = init_weights(UNetLike{I, bn}, wr, wu, seed = seed)
  state = init_state(UNetLike{I, bn})

  return UNetLike{I, bn}(weights, state)
end

