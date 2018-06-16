
# --------------------------------------------------------------------------- #
# Lessons, aka job descriptions for training processes

const Region      = NTuple{4, Int}
const ImageOrFile = Union{Image, String}
const LabelOrFile = Union{Label, String}
const Selection   = Tuple{ImageOrFile, LabelOrFile, Region}


# Remark: Images and labels can either be given as data or as path. 
# The selection region is *only* applied for data that is loaded via 
# a path. So if an image or a label is given as data, the selection 
# region is therefore ignored!
# In other words: If data is given, it is expected that the data
# was created with crop(data, region...)

mutable struct Lesson

  # Model
  model
  imgtype :: DataType
  kwargs  :: Vector{Any}

  # Algorithm
  optimizer :: String
  lr        :: Float64

  # Image and label data
  folder     :: String
  selections :: Array{Selection}

  # Data augmentation
  imageop :: ImageOp

  # Parameters
  epochs       :: Int
  batchsize    :: Int
  patchsize    :: Int
  patchmode    :: Int
  kernelsize   :: Int
  kernelheight :: Int

end


# --------------------------------------------------------------------------- #
# Lesson constructor

function Lesson(modelc;      # model or constructor function for a model
                imgtype      = RGBImage,
                kwargs       = Any[],
                folder       = "",
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

    return Lesson(modelc, imgtype, kwargs, optimizer, 
                  lr, folder, selections, imageop, epochs, 
                  batchsize, patchsize, patchmode,
                  kernelsize, kernelheight)
end


# --------------------------------------------------------------------------- #
# Training with lessons

function resolvesel(s, folder)
  img, lbl, region = s
  if typeof(img) == String
    img = crop(imgload(joinpath(folder, img)), region...)
  else
    # If data is given, expect that it was created by calling 
    # `crop` on the full image
    @assert imgsize(img) == region[4:-1:3]
  end
  if typeof(lbl) == String
    lbl = crop(lblload(joinpath(folder, img)), region...)
  else
    # If data is given, expect that it was created by calling 
    # `crop` on the full label
    @assert length(lbl) == length(crop(lbl, 1, 1, region[3:4]...))
  end
  return (img, lbl)
end

function train(lesson :: Lesson; kwargs...)
  if typeof(lesson.model) <: Model
    model = deepcopy(lesson.model)
  else
    model = lesson.model(lesson.imgtype; lesson.kwargs...)
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

