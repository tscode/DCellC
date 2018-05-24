
# --------------------------------------------------------------------------- #
# Abstract type of a synthetic Cell

abstract type Cell end

# --------------------------------------------------------------------------- #
# Simple implementation of a round synthetic Cell that has sharp boundaries

struct SharpCircleCell <: Cell
  radius :: Tuple{Float64, Float64}
  mag :: Tuple{Float64, Float64}
  SharpCircleCell(r, m) = new((r...), (m...))
end

SharpCircleCell(r::Real, m::Real = 1.) = SharpCircleCell((r, r), (m, m))
SharpCircleCell(r, m::Real = 1.) = SharpCircleCell((r...), (m, m))
SharpCircleCell(r::Real, m) = SharpCircleCell((r, r), (m...))

function (c::SharpCircleCell)()
  s = rand(2)
  r = s[1] * c.radius[1] + (1 - s[1]) * c.radius[2]
  m = s[2] * c.mag[1]    + (1 - s[2]) * c.mag[2]
  l = floor(Int, ceil(2 * r) / 2) * 2 + 1
  k = ceil(Int, l / 2)
  patch = zeros(l, l)

  for i in 1:l, j in 1:l
    patch[i, j] = (i - k)^2 + (j - k)^2 <= r^2 ? m : 0
  end

  return patch
end


# --------------------------------------------------------------------------- #
# Simple implementation of an elliptic Cell that has sharp boundaries

struct SharpEllipticCell <: Cell
  axes :: Tuple{Float64, Float64}
  mag :: Tuple{Float64, Float64}
  SharpEllipticCell(r, m) = new((r...), (m...))
end

SharpEllipticCell(r::Real, m::Real = 1.) = SharpEllipticCell((r, r), (m, m))
SharpEllipticCell(r, m::Real = 1.) = SharpEllipticCell((r...), (m, m))
SharpEllipticCell(r::Real, m) = SharpEllipticCell((r, r), (m...))

function (c::SharpEllipticCell)()
  s = rand(4)
  a = s[1] * c.axes[1] + (1 - s[1]) * c.axes[2]
  b = s[2] * c.axes[1] + (1 - s[2]) * c.axes[2]
  m = s[3] * c.mag[1]  + (1 - s[3]) * c.mag[2]
  t = s[4] * 2pi 

  l = floor(Int, ceil(2 * max(a, b)) / 2) * 2 + 1
  k = ceil(Int, l / 2)
  patch = zeros(l, l)

  for i in 1:l, j in 1:l
    v = [ cos(t) (-sin(t)); sin(t) cos(t) ] * [ i - k, j - k ]
    patch[i, j] = v[1]^2 / a^2 + v[2]^2 / b^2 <= 1 ? m : 0
  end

  return patch
end


# --------------------------------------------------------------------------- #
# Function that creates synthesized labeled images for provided Cell types

function synthesize( width, height, n :: Tuple{Integer, Integer}; 
                     cell = SharpCircleCell((5,10)), 
                     jitter :: Integer = 0, imgop = Id() )

  # Initialize image with background and prepare label
  image = zeros(Float32, width, height)

  n = rand(n[1]:n[2])
  label = zeros(Int, 2, n)
  
  # Iterate over all cell-realizations
  for i in 1:n
    c = cell()
    s = size(c)
    lx = floor(Int, s[1] / 2) + 1
    ly = floor(Int, s[2] / 2) + 1
    x, y = rand(lx:height-lx), rand(ly:width-ly)
    dx, dy = rand(-jitter:jitter, 2)
    label[:, i] = [x + dx, y + dy]
    xreg = x-lx+1 : x+lx-1
    yreg = y-ly+1 : y+ly-1
    image[yreg, xreg] += c
  end

  lbl = Label(label)
  img = GreyscaleImage(image)

  # Apply post-processing steps and return
  return apply(imgop, img, lbl)
end

