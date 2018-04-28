
module CellDC

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


end # CellDC
