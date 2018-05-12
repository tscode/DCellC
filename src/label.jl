
# --------------------------------------------------------------------------- #
# Labels -- a list of cell position coordinates

struct Label
  data :: Vector{Tuple{Int, Int}}
end

function Base.convert(::Type{Label}, mat::Matrix{Int})
  @assert (size(mat, 1) == 2)
  return Label([(mat[:,i]...) for i in 1:size(mat, 2)])
end

function Base.convert(::Type{Matrix{Int}}, lab::Label) 
  return hcat([[coords...] for coords in lab.data]...)
end

Label() = Label(Tuple{Int, Int}[])
Label(mat::Matrix{Int}) = convert(Label, mat)

# --------------------------------------------------------------------------- #
# See tuples of images and labels as "labeled Images"

const LabeledGreyscaleImage = Tuple{GreyscaleImage, Label}
const LabeledRGBImage = Tuple{RGBImage, Label}
const LabeledImage = Union{LabeledGreyscaleImage, LabeledRGBImage}


# --------------------------------------------------------------------------- #
# Proximity maps
# Proxymaps are "washed out" 2d arrays with peaks at given label positions

function proxymap(width, height, lbls :: Vector{Label}; 
                  kernelsize :: Int = 7, 
                  stddev :: Real = kernelsize/4,
                  peakheight :: Real = 100)

  # Create proximity maps from labels
  # Each point in the label gets converted to a gaussian activation

  n = length(lbls)

  # Pixel map
  prmap = zeros(Float32, width, height, 1, n)
  for i in 1:n 
    for (x, y) in lbls[i]
      prmap[y, x, 1, i] = 1.
    end
  end

  # Convolve the pixel-map with a suitable kernel
  c = ceil(Int, kernelsize/2)
  kernel = Float32[ exp(- ((i - c)^2 + (j - c)^2) / (2stddev^2)) 
                    for i in 1:kernelsize, j in 1:kernelsize ]
  kernel = reshape(kernel, size(kernel)..., 1, 1)

  # Scale the output such that deviations from real labels are
  # punished more than deviations from the background
  pad = floor(Int, kernelsize/2)
  prmap = reshape(conv4(kernel, prmap, padding=pad), width, height, n)

  return peakheight * prmap
end

function proxymap(width, height, lbl :: Label; kwargs...)
  return reshape(proxymap(width, height, [lbl]; kwargs...), width, height)
end


# --------------------------------------------------------------------------- #
# Adjacency
# Calculates a list of smallest distances between the spots in two labels

function adjacency(dlbl, tlbl)
  ddata, tdata = convert.(Matrix{Int}, (dlbl, tlbl))
  if isempty(ddata) && isempty(tdata)
    return 0.
  elseif isempty(ddata) || isempty(tdata)
    return Inf
  end

  n, m = length(tlbl), length(dlbl)
  vals = zeros(n + m)
  for i in 1:n
    dists = sqrt.(sum(abs2, ddata .- tdata[:,i], 1))[1,:]
    vals[i] = minimum(dists)
  end
  for i in 1:m
    dists = sqrt.(sum(abs2, ddata[:,i] .- tdata, 1))[1,:]
    vals[n+i] = minimum(dists)
  end
  return vals
end

function adjacency(model, img :: Image, lbl :: Label; kwargs...)
  return adjacency(label(model, img; kwargs...), lbl)
end


# --------------------------------------------------------------------------- #
# Convenience functions

Base.size(l :: Label, args...; kwargs...) = size(l.data, args..., kwargs...)
Base.length(l :: Label, args...; kwargs...) = length(l.data, args..., kwargs...)
Base.getindex(l :: Label, k :: Integer) = l.data[k]
Base.setindex!(l :: Label, v, k :: Integer) = setindex!(l.data, (v...), k)
Base.endof(l :: Label) = length(l)

Base.start(l::Label) = start(l.data)
Base.next(l::Label, state) = next(l.data, state)
Base.done(l::Label, state) = done(l.data, state)
Base.eltype(::Type{Label}) = Tuple{Int, Int}

Base.push!(l :: Label, coords) = push!(l.data, coords)
Base.pop!(l :: Label) = pop!(l.data)
Base.deleteat!(l :: Label, idx) = deleteat!(l.data, idx)


