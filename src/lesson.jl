
# --------------------------------------------------------------------------- #
# Lessons, aka job descriptions for training processes

const Region = Tuple{Int, Int, Int, Int}
const ImageOrFile = Union{Image, String}
const LabelOrFile = Union{Label, String}
const Selection = Tuple{ImageOrFile, LabelOrFile, Region}

struct Lesson

  # Model
  model
  imgtype    :: DataType
  batchnorm  :: Bool

  # Image and label data
  folder     :: String
  selections :: Array{Selection}

  # Data augmentation
  pipeline   :: Pipeline

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
                selections   = [],
                pipeline     = Pipeline(NoOp()),
                epochs       = 10,
                batchsize    = 1,
                patchsize    = 256,
                patchmode    = 0, 
                kernelsize   = 7,
                kernelheight = 100)

    return Lesson(modelc, imgtype, batchnorm,
                  folder, selections, pipeline, epochs, 
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

  lmgs = resolvesel.(lesson.selections, lesson.folder)

  # TODO: Pipelines!!
  train!(model, lmgs;
         epochs = lesson.epochs,
         batchsize  = lesson.batchsize,
         patchsize  = lesson.patchsize,
         patchmode  = lesson.patchmode,
         kernelsize = lesson.kernelsize,
         peakheight = lesson.kernelheight,
         shuffle = true,
         kwargs...)

  return model 
end

