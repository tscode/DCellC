
# --------------------------------------------------------------------------- #
# Operations and pipelines

abstract type ImageOp end

struct Pipeline{N}
  ops :: NTuple{N, ImageOp}
end

Pipeline(op::ImageOp) = Pipeline((op,))

import Base.|>
|>(op1::ImageOp, op2::ImageOp) = Pipeline((o1, o2))
|>(pip::Pipeline, op::ImageOp) = Pipeline((pip.ops..., op))

# --------------------------------------------------------------------------- #
# Applications of pipelines

function apply(img :: Image, pip :: Pipeline)
  for op in pip.ops
    img = apply(img, op)
  end
  return img
end

function apply(img :: Image, lbl :: Label, pip :: Pipeline)
  for op in pip.ops
    img, lbl = apply(img, op)
  end
  return img, lbl
end

apply(img :: Image, op :: ImageOp) = apply(img, Label(), op)[1]

function apply(imgop :: Tuple{Image, ImageOp}, args...; kwargs...) 
  apply(imgop[1], imgop[2], args...; kwargs...)
end

# --------------------------------------------------------------------------- #
# Identity operation

struct NoOp <: ImageOp end

apply(img :: Image, lbl :: Label, :: NoOp) = img, lbl

# --------------------------------------------------------------------------- #
# Flips along vertical and horizontal axis

struct Flip <: ImageOp 
  prob :: Tuple{Float64, Float64}
end

Flip(;vertical = 0.5, horizontal = 0.5) = Flip((vertical, horizontal))

function apply{I <: Image}(img :: I, lbl :: Label, op::Flip)

  n = length(lbl)

  # Vertical flip
  if rand() <= op.prob[1]
    l = convert(Matrix{Int}, lbl)
    img = I(flipdim(img.data, 1))
    dat = [ [l[1,i], size(img, 1) - l[2,i] + 1] for i in 1:n ]
    lbl = Label(hcat(dat...))
  end

  # Horizontal flip
  if rand() <= op.prob[2]
    l = convert(Matrix{Int}, lbl)
    img = I(flipdim(img.data, 2))
    dat = [ [size(img, 2) - l[1,i] + 1, l[2,i]] for i in 1:n ]
    lbl = Label(hcat(dat...))
  end

  return img, lbl
end

# --------------------------------------------------------------------------- #
# Softening the image with gaussian kernel

struct Soften <: ImageOp
  kernelsize :: Int
end

Soften(; kernelsize = 3) = Soften(kernelsize)

function apply{I <: Image}(img :: I, lbl :: Label, op :: Soften)
  return I(imfilter(data(img), Kernel.gaussian(op.kernelsize))), lbl
end

# --------------------------------------------------------------------------- #
# Add pixelwise, independent noise

struct PixelNoise <: ImageOp
  source :: Function
  amplitude :: Float64
end

PixelNoise(; source = rand, amplitude = 0.2) = PixelNoise(source, amplitude)

function apply{I <: Image}(img :: I, lbl :: Label, op :: PixelNoise)
  return I(data(img) + op.amplitude * op.source(size(img))), lbl
end


# --------------------------------------------------------------------------- #
# TODO: Rotation, Resize, Shear, Crop, ...




