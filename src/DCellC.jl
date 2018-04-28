
module DCellC

# --------------------------------------------------------------------------- #
# Import external functionality

using Knet

import Colors
import ImageFiltering
import JLD2


# --------------------------------------------------------------------------- #
# Feature library

include("image.jl")
include("label.jl")
include("imageop.jl")
include("synthetic.jl")
include("model.jl")
include("training.jl")
include("util.jl")

# --------------------------------------------------------------------------- #
# Exports

export Image, GreyscaleImage, RGBImage, 
       ordered_patches, random_patches, 
       imgdata, imgsize, imgchannels

export Label, adjacency, proxymap

export ImageOp, Pipeline, apply,
       NoOp, Flip, Soften, PixelNoise

export Model, FCModel, Unetlike, Multiscale3, FCRNA,
       weights, state, save, load, density, label

export train!, loss



end # DCellC
