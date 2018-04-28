

# --------------------------------------------------------------------------- #
# General image type

abstract type Image end

# --------------------------------------------------------------------------- #
# Greyscale Images

struct GreyscaleImage <: Image
  data :: Array{Float32, 2} 
end

# --------------------------------------------------------------------------- #
# RGB Images

struct RGBImage <: Image
  data :: Array{Float32, 3} 
end


# --------------------------------------------------------------------------- #
# Create datasets from high-res images

function intervals(L, l, m)
  grid = Int[]
  i = 1

  while i + l - 1 <= L
    push!(grid, i)
    i += l + m
  end

  return grid
end


function ordered_patches{I <: Image}(img :: I, lbl; 
                                     size = (128, 128), 
                                     offset = nothing,
                                     margin :: Integer = 0,
                                     shuffle :: Bool = true,
                                     imageop = NoOp(),
                                     multitude = 1)

  # Note: components of argument size should be divisible by 8,
  # otherwise some (multiresolution) networks will not like the input

  n, m = Base.size(img)[1:2]

  @assert (n >= size[1] && m >= size[2])
  @assert (-margin < min(size...))

  ygrid = intervals(n, size[1], margin)
  xgrid = intervals(m, size[2], margin)

  if offset == nothing
    dy = div(n - (ygrid[end] + size[1] - 1), 2)
    dx = div(m - (xgrid[end] + size[2] - 1), 2)
    offset = (dy, dx)
  end

  ygrid = Int[ i + offset[1] for i in ygrid ]
  xgrid = Int[ j + offset[2] for j in xgrid ]

  imgs = I[]
  lbls = Label[]

  for i in ygrid, j in xgrid
    ii = i + size[1] - 1
    jj = j + size[2] - 1
    sel = (i .<= lbl.data[2,:] .<= ii) .& (j .<= lbl.data[1,:] .<= jj)
    push!(imgs, I(img.data[i:ii, j:jj]))
    push!(lbls, Label(lbl.data[:,sel] .- [j-1, i-1]))
  end

  data = vcat( (apply.(imgs, lbls, imageop) for i in 1:multitude)... )
  imgs = [ img for (img, _) in data ]
  lbls = [ lbl for (_, lbl) in data ]

  return imgs, lbls
end


function random_patches{I <: Image}(img :: I, number; 
                                    size = (128, 128), 
                                    aug = Function[])

  # Note: components of argument size should be divisible by 8,
  # otherwise some (multiresolution) networks will not like the input

  n, m = Base.size(img)[1:2]
  @assert (n >= size[1] && m >= size[2])

  imgs = I[]
  lbls = Label[]

  for k in 1:number
    i, j = rand(1:n-size[1]), rand(1:m-size[2])
    ii = i + size[1] - 1
    jj = j + size[2] - 1
    sel = (i .<= lbl.data[2,:] .<= ii) .& (j .<= lbl.data[1,:] .<= jj)
    push!(imgs, I(img.data[i:ii, j:jj]))
    push!(lbls, Label(lbl.data[:,sel] .- [j-1, i-1]))
  end

  data = vcat( (apply.(imgs, lbls, imageop) for i in 1:multitude)... )
  imgs = [ img for (img, _) in data ]
  lbls = [ lbl for (_, lbl) in data ]

  return imgs, lbls
end


ordered_patches(img; kwargs...) = ordered_patches(img, Label(); kwargs...)[1]
random_patches(img; kwargs...) = random_patches(img, Label(); kwargs...)[1]


# --------------------------------------------------------------------------- #
# Convenience functions

imgdata(img :: Image) = img.data
imgsize(a) = size(a)[1:2]

imgchannels(::GreyscaleImage) = 1
imgchannels(::RGBImage) = 3
imgchannels(::Type{<: GreyscaleImage}) = 1
imgchannels(::Type{<: RGBImage}) = 1

#Base.convert{A}(::Type{A}, img::GreyscaleImage) = convert(A, img.data)
Base.size(img::Image, args...; kwargs...) = size(img.data, args..., kwargs...)

