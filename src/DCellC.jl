
module DCellC

# --------------------------------------------------------------------------- #
# Import external functionality

using Knet

import Colors
import FileIO
import Images
import ImageFiltering
import JLD2


# --------------------------------------------------------------------------- #
# Feature library

include("image.jl")
include("label.jl")
include("imageop.jl")

include("model.jl")
include("training.jl")
include("lesson.jl")

include("synthetic.jl")
include("io.jl")
include("util.jl")

# --------------------------------------------------------------------------- #
# Exports

export Image, GreyscaleImage, RGBImage, 
       ordered_patches, random_patches,
       imgdata, imgsize, imgchannels, imgconvert, 
       imgtype, crop, greyscale

export Label, adjacency, proxymap

export ImageOp, RandomImageOp, Pipeline, 
       Id, FlipV, FlipH, Jitter, Soften, 
       PixelNoise, StretchV, StretchH,
       apply

export Model, FCModel, UNetLike, Multiscale3, FCRNA,
       weights, state, save, load, density, label,
       density_patched, hasbatchnorm 

export train!, train, loss

export Lesson, Selection, lessonsave, lessonload

# legacy stuff
export SharpCircleCell, SharpEllipticCell, 
       synthesize, PP

export fileext, imgsave, imgload, lblsave, lblload,
       lmgsave, lmgload, modelsave, modelload

end # DCellC
