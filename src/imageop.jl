
# --------------------------------------------------------------------------- #
# Image Operations and, more general, random Image Operations

abstract type ImageOp end


# --------------------------------------------------------------------------- #
# Fallback apply methods

apply(o :: ImageOp, img :: Image) = apply(o, img, Label())[1]
apply(o :: ImageOp, lmg :: LabeledImage) = apply(o, lmg...)

# --------------------------------------------------------------------------- #
# Identity operation

struct Id <: ImageOp end

apply(:: Id, img :: Image, lbl :: Label) = (img, lbl)

# --------------------------------------------------------------------------- #
# Random operations - randomly choose one of several possible ops

struct RandomImageOp{N} <: ImageOp
  options :: NTuple{N, ImageOp}
  freqs :: NTuple{N, Float64}

  function RandomImageOp{N}(opts, freqs) where {N}
    return new(opts, clamp.(freqs, 0, Inf))
  end
end


function RandomImageOp(options, freqs) 
  N = length(options)
  return RandomImageOp{N}(tuple(options...), tuple(freqs...))
end

RandomImageOp(o :: ImageOp, f :: Real) = RandomImageOp((o, Id()), (f, 1))

function apply(o :: RandomImageOp, img :: Image, lbl :: Label)
  r = rand()
  p = collect(o.freqs ./ sum(o.freqs))
  k = findfirst(x -> x > r, cumsum(p))
  apply(o.options[k], img, lbl)
end

import Base.+

addone(o :: ImageOp) = (o, 1.)
addone(o :: Tuple{ImageOp, Real}) = (o[1], float(o[2]))

function +(o :: Union{ImageOp, Tuple{ImageOp, Real}}, os...)
  os = addone.([o; os...])
  return RandomImageOp((first.(os)...), (second.(os)...))
end

# --------------------------------------------------------------------------- #
# Processing Pipeline - operations in sequence

struct Pipeline{N} <: ImageOp
  ops   :: NTuple{N, ImageOp}
end

function apply(pip :: Pipeline, img :: Image, lbl :: Label)
  for op in pip.ops
    img, lbl = apply(op, img, lbl)
  end
  return img, lbl
end

import Base.*

mkrandom(o :: ImageOp) = o

function mkrandom(o :: Tuple{ImageOp, Real}) 
  return RandomImageOp((o[1], Id()), (o[2], 1.))
end

function *(o :: Union{ImageOp, Tuple{ImageOp, Real}}, os...)
  os = (mkrandom.([o; os...]) ...)
  return Pipeline(os)
end


# --------------------------------------------------------------------------- #
# Parsing Image Operations

function Base.parse(::Type{ImageOp}, s::String)
  return eval(parse(s)) 
end

# --------------------------------------------------------------------------- #
# Flips along vertical and horizontal axis

struct FlipV <: ImageOp end
struct FlipH <: ImageOp end

function apply{I <: Image}(:: FlipV, img :: I, lbl :: Label)
  img = I(flipdim(imgdata(img), 1))
  lbl = Label([ (x, size(img, 1) - y + 1) for (x,y) in lbl ])
  return img, lbl
end

function apply{I <: Image}(:: FlipH, img :: I, lbl :: Label)
  img = I(flipdim(imgdata(img), 2))
  lbl = Label([ (size(img, 2) - x + 1, y) for (x,y) in lbl ])
  return img, lbl
end

MaybeFlip(fv = 1, fh = fv) = (FlipV(), fv) * (FlipH(), fh)

# --------------------------------------------------------------------------- #
# Jitter -- move the label spots randomly without effecting the image

struct Jitter <: ImageOp 
  d :: Int
end

Jitter() = Jitter(3)

function apply{I <: Image}(o :: Jitter, img :: I, lbl :: Label)
  lbl = Label([c .+ (rand(-o.d:o.d, 2)...) for c in lbl])
  return img, crop(lbl, 1, 1, imgsize(img)...)
end

# --------------------------------------------------------------------------- #
# Softening the image with gaussian kernel

struct Soften <: ImageOp
  r :: Int
end

Soften() = Soften(3)

function apply{I <: Image}(o :: Soften, img :: I, lbl :: Label)
  data = copy(imgdata(img)[:,:,:])
  for c in 1:imgchannels(img)
    data[:,:,c] = ImageFiltering.imfilter(imgdata(img)[:, :, c], 
                                          Images.Kernel.gaussian(o.r))
  end
  return I(data), lbl
end

# --------------------------------------------------------------------------- #
# Add pixelwise, independent noise

struct PixelNoise <: ImageOp
  amp :: Float64
  source :: Function
end

PixelNoise(amp = 0.1; source = rand) = PixelNoise(amp, source)

function apply{I <: Image}(o :: PixelNoise, img :: I, lbl :: Label)
  return I(imgdata(img) + o.amp * o.source(size(img))), lbl
end


# --------------------------------------------------------------------------- #
# Stretch Image in vertical or horizontal direction

struct StretchV <: ImageOp
  f :: Float64
  StretchV(f) = new(clamp(f, 1, Inf))
end

struct StretchH <: ImageOp
  f :: Float64
  StretchH(f) = new(clamp(f, 1, Inf))
end

StretchV() = StretchV(1.1)
StretchH() = StretchH(1.1)

function apply{I <: Image}(o :: StretchV, img :: I, lbl :: Label)
  
  # Prepare
  n, m = imgsize(img)
  n2 = round(Int, o.f*n)
  a = div((n2 - n), 2) + 1

  # Stretch and crop data
  data = copy(imgdata(img)[:,:,:])
  for c in 1:imgchannels(img)
    data[:,:,c] = crop(Images.imresize(data[:,:,c], n2, m), 1, a, m, n)
  end

  # Also stretch and crop label
  lbl = Label([(x, round(Int, o.f*y)) for (x,y) in lbl])
  lbl = crop(lbl, 1, a, m, n)

  return I(data), lbl
end


function apply{I <: Image}(o :: StretchH, img :: I, lbl :: Label)
  
  # Prepare
  n, m = imgsize(img)
  m2 = round(Int, o.f*m)
  a = div((m2 - m), 2) + 1

  # Stretch and crop data
  data = copy(imgdata(img)[:,:,:])
  for c in 1:imgchannels(img)
    data[:,:,c] = crop(Images.imresize(data[:,:,c], n, m2), a, 1, m, n)
  end

  # Also stretch and crop label
  lbl = Label([(round(Int, o.f*x), y) for (x,y) in lbl])
  lbl = crop(lbl, a, 1, m, n)

  return I(data), lbl
end


Stretch(f = 1.1) = StretchV(f) * StretchH(f)


# --------------------------------------------------------------------------- #
# Gamma scaling

struct Gamma <: ImageOp
  gamma :: Float64
end

function apply{I <: Image}(o :: Gamma, img :: I, lbl :: Label)
  return I(imgdata(data).^(1/o.gamma)), lbl
end

# --------------------------------------------------------------------------- #
# Background offset

struct Offset <: ImageOp
  c :: NTuple{3, Float64}
end

Offset(c :: Float64) = Offset((c,c,c))

function apply(o :: Offset, img :: GreyscaleImage, lbl :: Label)
  return GreyscaleImage(imgdata(img) + mean(o.c)), lbl
end

function apply(o :: Offset, img :: RGBImage, lbl :: Label)
  return RGBImage(imgdata(img) .+ reshape([o.c...], (1,1,3))), lbl
end

# --------------------------------------------------------------------------- #
# TODO: Rotation, Shear, Crop...
# TODO: HSV wiggling for RGB images!



