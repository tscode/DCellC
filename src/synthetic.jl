
# --------------------------------------------------------------------------- #
# Abstract type of a synthetic Cell

abstract type Cell end

# --------------------------------------------------------------------------- #
# Simple implementation of a round synthetic Cell that has sharp boundaries

struct SharpCircleCell <: Cell
  radius :: Tuple{Float64, Float64}
  mag :: Tuple{Float64, Float64}
end

SharpCircleCell(r::Real, m::Real = 1.) = SharpCircleCell((r, r), (m, m))
SharpCircleCell(r, m::Real = 1.) = SharpCircleCell((r...), (m, m))
SharpCircleCell(r::Real, m) = SharpCircleCell((r, r), (m...))
SharpCircleCell(r, m) = SharpCircleCell((r...), (m...))

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
end

SharpEllipticCell(r::Real, m::Real = 1.) = SharpEllipticCell((r, r), (m, m))
SharpEllipticCell(r, m::Real = 1.) = SharpEllipticCell((r...), (m, m))
SharpEllipticCell(r::Real, m = 1.) = SharpEllipticCell((r, r), (m...))
SharpEllipticCell(r, m = 1.) = SharpEllipticCell((r...), (m...))

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
                     jitter :: Integer = 0, pp = nothing )

  # Initialize image with background and prepare label
  image = zeros(width, height)

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

  # Apply post-processing steps
  if pp != nothing
    if !applicable(start, pp)
      pp = [ pp ]
    end
    for p in pp
      image = p(image)
    end
  end

  # Return the final image and the corresponding label
  return GreyscaleImage(image), Label(label)
end


# --------------------------------------------------------------------------- #
# Post-Processing functionality

module PP

  using ImageFiltering

  # Simple Gaussian smoothing
  function soften(size = 3)
    return img ->
      imfilter(img, Kernel.gaussian(size))
  end

  # Add pixelwise noise to the image
  function noise(amp = 0.2, f = rand)
    return img ->
      img + amp * f(size(img)...)
  end


  # Add background gradients 
  #=function gradient(min, max; unit = nothing)=#
    #=return function f(img)=#
      #=d = sqrt(sum(abs2, (size(img))))=#
      #=t = rand() * 2pi=#
      #=unit = (unit == nothing) ? norm(size(img)...) : unit=#
      #=start = =#
      #=grad(i, j) = =#
    #=end=#
  #=end=#

end
