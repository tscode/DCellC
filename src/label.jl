
# --------------------------------------------------------------------------- #
# Labels -- a list of cell position coordinates

struct Label
  data :: Matrix{Int}
  function Label(data)
    @assert (size(data, 1) == 2)
    new(data)
  end
end

Label() = Label(zeros(Int, 2, 0))

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
    label = lbls[i]
    for j in 1:size(label.data, 2)
      x, y = label.data[:, j]
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
  if isempty(dlbl.data) && isempty(tlbl.data)
    return 0.
  elseif isempty(dlbl.data) || isempty(tlbl.data)
    return Inf
  end

  n, m = length(tlbl), length(dlbl)
  vals = zeros(n + m)
  for i in 1:n
    dists = sqrt.(sum(abs2, dlbl.data .- tlbl.data[:,i], 1))[1,:]
    vals[i] = minimum(dists)
  end
  for i in 1:m
    dists = sqrt.(sum(abs2, dlbl.data[:,i] .- tlbl.data, 1))[1,:]
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
Base.length(l :: Label, args...; kwargs...) = div(length(l.data, args..., kwargs...), 2)
Base.getindex(l :: Label, k :: Integer) = l.data[:,k]

