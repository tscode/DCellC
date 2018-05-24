

# --------------------------------------------------------------------------- #
# General image type

abstract type Image end

# --------------------------------------------------------------------------- #
# Greyscale Images

struct GreyscaleImage <: Image
  data :: Array{Float32, 2} 
  GreyscaleImage(data) = new(autoscale(convert(Array{Float32}, data)))
end

function GreyscaleImage(data :: Array{Float32, 3})
  @assert size(data, 3) == 1
  GreyscaleImage(reshape(data, size(data)[1:2]))
end

# --------------------------------------------------------------------------- #
# RGB Images

struct RGBImage <: Image
  data :: Array{Float32, 3} 
  RGBImage(data) = new(autoscale(convert(Array{Float32}, data)))
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
                                     imageop = Id(),
                                     multitude = 1)

  # Note: components of argument size should be divisible by 8,
  # otherwise some (multiresolution) networks will not like the input

  n, m = imgsize(img)

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
  ld = convert(Matrix{Int}, lbl)

  for i in ygrid, j in xgrid
    ii = i + size[1] - 1
    jj = j + size[2] - 1
    push!(imgs, I(img.data[i:ii, j:jj, :]))
    if !isempty(lbl)
      sel = (i .<= ld[2,:] .<= ii) .& (j .<= ld[1,:] .<= jj)
      push!(lbls, Label(ld[:,sel] .- [j-1, i-1]))
    else
      push!(lbls, Label())
    end
  end

  data = vcat( (apply.(imageop, imgs, lbls) for i in 1:multitude)... )
  imgs = [ img for (img, _) in data ]
  lbls = [ lbl for (_, lbl) in data ]

  return imgs, lbls
end


function random_patches{I <: Image}(img :: I, lbl, number; 
                                    size = (128, 128), 
                                    imageop = Id(),
                                    multitude = 1)

  # Note: components of argument size should be divisible by 8,
  # otherwise some (multiresolution) networks will not like the input

  n, m = imgsize(img)
  @assert (n >= size[1] && m >= size[2])

  imgs = I[]
  lbls = Label[]
  ld = convert(Matrix{Int}, lbl)

  for k in 1:number
    i, j = rand(1:n-size[1]), rand(1:m-size[2])
    ii = i + size[1] - 1
    jj = j + size[2] - 1
    push!(imgs, I(img.data[i:ii, j:jj, :]))
    if !isempty(lbl)
      sel = (i .<= ld[2,:] .<= ii) .& (j .<= ld[1,:] .<= jj)
      push!(lbls, Label(ld[:,sel] .- [j-1, i-1]))
    else
      push!(lbls, Label())
    end
  end

  data = vcat( (apply.(imageop, imgs, lbls) for i in 1:multitude)... )
  imgs = [ img for (img, _) in data ]
  lbls = [ lbl for (_, lbl) in data ]

  return imgs, lbls
end


function ordered_patches(img; kwargs...) 
  return ordered_patches(img, Label(); kwargs...)[1]
end

function random_patches(img, number; kwargs...)
  return random_patches(img, Label(), number; kwargs...)[1]
end


# --------------------------------------------------------------------------- #
# Convenience functions

imgdata(img :: Image) = img.data
imgsize(a) = size(a)[1:2]

imgchannels(::GreyscaleImage) = 1
imgchannels(::RGBImage) = 3
imgchannels(::Type{<: GreyscaleImage}) = 1
imgchannels(::Type{<: RGBImage}) = 3

Base.size(img::Image, args...; kwargs...) = size(img.data, args..., kwargs...)

function imgconvert{T <: Images.Gray}(imgarray :: Array{T})
  data = permutedims(Images.channelview(imgarray), (2, 3, 1))
  return GreyscaleImage(reshape(data, size(data)[1:2]...))
end

function imgconvert{T <: Images.RGB}(imgarray :: Array{T})
  data = permutedims(Images.channelview(imgarray), (2, 3, 1))
  return RGBImage(data)
end

function crop(data :: Matrix{Float32}, x :: Int, y :: Int, w :: Int, h :: Int)
  return data[y:y+h-1,x:x+w-1]
end

function crop(img :: RGBImage, x :: Int, y :: Int, w :: Int, h :: Int)
  return RGBImage(imgdata(img)[y:y+h-1,x:x+w-1,:])
end

function crop(img :: GreyscaleImage, x :: Int, y :: Int, w :: Int, h :: Int)
  return GreyscaleImage(imgdata(img)[y:y+h-1,x:x+w-1])
end

Base.similar(img::GreyscaleImage) = GreyscaleImage(zeros(Float32, imsize(img)))


