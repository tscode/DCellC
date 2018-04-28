
# --------------------------------------------------------------------------- #
# The "U-Net-like" model from (Pen, Yang, Li, et al.; 2018)

struct UNetLike{I <: Image, BN} <: Model
  weights :: Array
end


# --------------------------------------------------------------------------- #
# Network architecture

# Need this helper function that uses the raw, unwrapped weights
# Otherwise automatic differentiation complains
function density(w, x, ::Type{UNetLike})
  y1 = rconv(w[2], rconv(w[1], x))
  y2 = rconv(w[4], rconv(w[3], pool(y1)))
  y3 = rconv(w[6], rconv(w[5], pool(y2)))
  y4 = rconv(w[8], rconv(w[7], pool(y3)))

  z3 = rconv(w[11], rconv(w[10], uconv(w[9], y3, y4)))
  z2 = rconv(w[14], rconv(w[13], uconv(w[12], y2, z3))) 
  z1 = rconv(w[17], rconv(w[16], uconv(w[15], y1, z2)))

  return conv4(w[18], z1, padding=1)[:,:,1,:]
end

function density(m :: UNetLike, img :: GreyscaleImage)
  x = reshape(img.data, (size(img.data)..., 1, 1))
  x = convert(Array{Float32}, x)
  w = m.weights
  return density(w, x, UNetLike)[:,:,1]
end


# --------------------------------------------------------------------------- #
# Initialization and construction

# Convolution kernel
wr(i, o) = gaussian(Float32, 3, 3, i, o, std=0.060)

# Up-convolution kernel
wu(i, o) = bilinear(Float32, 2, 2, o, i)

# Constructor
function UNetLike(; seed :: Integer = rand(1:10000))
  Knet.setseed(seed)

  weights = Any[
    wr(1, 64),  wr(64, 64), wr(64, 64), wr(64, 64), 
    wr(64, 64), wr(64, 64), wr(64, 64), wr(64, 64),
    wu(64, 64), wr(128, 64), wr(64, 64), wu(64, 64),
    wr(128, 64), wr(64, 64), wu(64, 64), wr(128, 64),
    wr(64, 64), wr(64, 1)
  ]

  return UNetLike(weights)
end



# --------------------------------------------------------------------------- #
# Model interface functions 

weights(m :: UNetLike) = m.weights
state(m :: UNetLike) = m.weights

