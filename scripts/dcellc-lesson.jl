
function args_lesson(s)
  @add_arg_table s begin
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
      default = 5e-5
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
  if args["learning-rate"] != -1
    lesson.lr = args["learning-rate"]
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
