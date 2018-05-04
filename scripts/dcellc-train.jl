
function args_train(s)
  @add_arg_table s begin
    "--model-type", "-m"
      help = "model to be used - one of 'unetlike', 'multiscale3', 'fcrna'"
      arg_type = String
      default = "multiscale3"
    "--patch-number", "-n"
      help = "number of patches used for training"
      arg_type = Int
      default = 10
    "--patch-size", "-s"
      help = "number of patches used for training"
      arg_type = Int
      default = 10
    "--test-size", "--testset-size", "-t"
      help = "number of patches used for testing"
      arg_type = Int
      default = 5
    "--batch-size", "-b"
      help = "number of patches per mini-batch during training"
      arg_type = Int
      default = 1
    "--batch-normalization", "-B"
      help = "whether to activate batch normalization"
      action = :store_true
    "--no-gpu"
      help = "prevent usage of gpu acceleration even if gpu support is detected"
      action = :store_true
    "--epochs", "-e"
      help = "number of training epochs"
      arg_type = Int
      default = 10
    "--learning-rate", "-l"
      help = "initial learning rate for the chosen optimizer [-o]"
      arg_type = Float64
      default = 5e-5
    "--noise", "-N"
      help = "magnitude of the applied random noise on the patches (TODO)"
      arg_type = Float64
      default = 0.2
    "--proximity-kernel-size", "-k"
      help = "size of the peaks in the proximity map"
      arg_type = Int
      default = 7
    "--proximity-kernel-height", "-K"
      help = "height of the peaks in the proximity map"
      arg_type = Float64
      default = 100.
    "--optimizer", "-o"
      help = "optimizer used - one of 'adam', 'rmsprop', 'nesterov'"
      arg_type = String
      default = "adam"
    "model"
      required = true
      arg_type = String
      help = "DCellC model (.dccm file) to save the model in after training"
    "images"
      nargs='+'
      required = true
      arg_type = String
      help = "image (and corresponding annotation) files that the model trained on"
  end
end


function cmd_train(args)

  # Load labeled images

  lmgs = [ lmgload(imgfile) for imgfile in args["images"] ]

  # Make sure that all images have the same type

  I = typeof(first(lmgs[1]))

  if !all(typeof.(first.(lmgs)) .== I) 
    println("Error: all image files must have the same color format")
    exit()
  end

  # Extract some more options

  mtype = args["model-type"]
  otype = args["optimizer"]
  bn    = args["batch-normalization"]

  # Initialize the model

  if mtype == "unetlike"
    model = UNetLike(I, bn=bn)
  elseif mtype == "multiscale3"
    model = Multiscale3(I, bn=bn)
  elseif mtype == "fcrna"
    model = FCRNA(I, bn=bn)
  end


  if otype == "adam"
    opt = Adam
  elseif otype == "rmsprop"
    opt = Rmsprop
  elseif otype == "nesterov"
    opt = Nesterov
  else
    println(STDERR, "Optimizer $(args["optimizer"]) invalid")
  end

  if args["no-gpu"] || gpu() < 0
    println("# CPU mode")
    at = Array{Float32}
  else
    println("# GPU mode")
    at = KnetArray{Float32}
  end

  train!(model, lmgs, 
         epochs = args["epochs"],
         modelpath = args["model"],
         patchmode = args["patch-number"],
         patchsize = args["patch-size"],
         batchsize = args["batch-size"],
         peakheight = args["proximity-kernel-height"],
         kernelsize = args["proximity-kernel-size"],
         record = false, at = at,
         lr = args["learning-rate"])
end
