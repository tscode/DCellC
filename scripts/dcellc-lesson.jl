
function args_lesson(s)
  @add_arg_table s begin
    "--model-type", "-m"
      help = "model to be used - one of 'unetlike', 'multiscale3', 'fcrna'"
      arg_type = String
      default = "multiscale3"
    "--patch-mode", "-n"
      help = "number of patches used for training"
      default = nothing
    "--patch-size", "-s"
      help = "size of the patches used for training"
      arg_type = Int
      default = -1
    "--batch-size", "-b"
      help = "number of patches per mini-batch during training"
      arg_type = Int
      default = -1
    "--batch-normalization", "-B"
      help = "whether to activate batch normalization (0 or 1)"
      arg_type = Int
      default = -1
    "--directory", "-d"
      help = "Overwrite image/label folder suggested by the lesson file"
      arg_type = String
      default = ""
    "--optimizer", "-o"
      help = "Overwrite optimizer suggested by the lesson file"
      arg_type = String
      default = ""
    "--epochs", "-e"
      help = "number of training epochs"
      arg_type = Int
      default = -1
    "--learning-rate", "-l"
      help = "initial learning rate for the chosen optimizer [-o]"
      arg_type = Float64
      default = -1.
    "--proximity-kernel-size", "-k"
      help = "size of the peaks in the proximity map"
      arg_type = Int
      default = -1
    "--proximity-kernel-height", "-K"
      help = "height of the peaks in the proximity map"
      arg_type = Float64
      default = -1.
    "--no-gpu"
      help = "prevent usage of gpu acceleration even if gpu support is detected"
      action = :store_true
    "lesson"
      help = "DCellC lesson (.dcct) file used for training"
      required = true
      arg_type = String
    "modelfile"
      help = "DCellC model (.dccm) file that the trained model is saved to"
      required = true
      arg_type = String
  end
end

function cmd_lesson(args)
  lesson = lessonload(args["lesson"])

  if args["directory"] != ""
    lesson.folder = args["directory"]
  end
  if args["optimizer"] != ""
    lesson.optimizer = args["optimizer"]
  end
  if args["epochs"] != -1
    lesson.epochs = args["epochs"]
  end
  if args["learning-rate"] > 0
    lesson.lr = args["learning-rate"]
  end
  if args["patch-mode"] != nothing
    lesson.patchmode = args["patch-mode"]
  end
  if args["patch-size"] != -1
    lesson.patchsize = args["patch-mode"]
  end
  if args["batch-size"] != -1
    lesson.batchsize = args["batch-mode"]
  end
  if args["batch-normalization"] != -1
    lesson.batchnorm = Bool(args["batch-normalization"])
  end
  if args["proximity-kernel-size"] != -1
    lesson.kernelsize = Bool(args["proximity-kernel-size"])
  end
  if args["proximity-kernel-height"] > 0
    lesson.kernelheight = Bool(args["proximity-kernel-height"])
  end

  if args["no-gpu"] || gpu() < 0
    println("# CPU mode")
    at = Array{Float32}
  else
    println("# GPU mode")
    at = KnetArray{Float32}
  end

  train(lesson, modelpath = args["modelfile"], record = false)
end

