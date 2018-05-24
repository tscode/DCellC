
# --------------------------------------------------------------------------- #
# Lessons, aka job descriptions for training processes

const Region = Tuple{Int, Int, Int, Int}
const ImageOrFile = Union{Image, String}
const LabelOrFile = Union{Label, String}
const Selection = Tuple{ImageOrFile, LabelOrFile, Region}

mutable struct Lesson

  # Model
  model
  imgtype    :: DataType
  batchnorm  :: Bool

  # Algorithm
  optimizer :: String
  lr :: Float64

  # Image and label data
  folder     :: String
  selections :: Array{Selection}

  # Data augmentation
  imageop   :: ImageOp

  # Parameters
  epochs       :: Int
  batchsize    :: Bool 
  patchsize    :: Int
  patchmode    :: Int
  kernelsize   :: Int
  kernelheight :: Int

end


# --------------------------------------------------------------------------- #
# Lesson constructor

function Lesson(modelc;      # model or constructor function for a model
                folder       = "",
                imgtype      = RGBImage,
                batchnorm    = true,
                optimizer    = "adam", # allowed: adam, rmsprop, nesterov
                lr           = 1e-4,
                selections   = [],
                imageop      = Id(),
                epochs       = 10,
                batchsize    = 1,
                patchsize    = 256,
                patchmode    = 0, 
                kernelsize   = 7,
                kernelheight = 100)

    return Lesson(modelc, imgtype, batchnorm, optimizer, 
                  lr, folder, selections, imageop, epochs, 
                  batchsize, patchsize, patchmode,
                  kernelsize, kernelheight)
end


# --------------------------------------------------------------------------- #
# Training with lessons

function resolvesel(s, folder)
  img, lbl, sel = s
  if typeof(img) == String
    img = imgload(joinpath(folder, img))
  end
  if typeof(lbl) == String
    lbl = lblload(joinpath(folder, img))
  end
  return crop((img, lbl), sel...)
end

function train(lesson :: Lesson; kwargs...)
  if typeof(lesson.model) <: Model
    model = deepcopy(lesson.model)
  else
    model = lesson.model(lesson.imgtype, bn=lesson.batchnorm)
  end

  if lesson.optimizer == "adam"
    opt = Knet.Adam
  elseif lesson.optimizer == "rmsprop"
    opt = Knet.Rmsprop
  elseif lesson.optimizer == "nesterov"
    opt = Knet.Nesterov
  else
    warn("Optimizer $optimizer not valid. Using fallback optimizer adam")
    opt = Knet.Adam
  end

  lmgs = resolvesel.(lesson.selections, lesson.folder)

  train!(model, lmgs;
         epochs     = lesson.epochs,
         batchsize  = lesson.batchsize,
         patchsize  = lesson.patchsize,
         patchmode  = lesson.patchmode,
         kernelsize = lesson.kernelsize,
         peakheight = lesson.kernelheight,
         imageop    = lesson.imageop,
         lr         = lesson.lr,
         opt        = opt, 
         shuffle    = true, 
         kwargs...)

  return model 
end

