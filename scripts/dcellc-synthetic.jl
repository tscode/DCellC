#! /usr/bin/env julia

# ---------------------------------------------------------------------------- #
# Parsing of command line arguments

using ArgParse

function ArgParse.parse_item{R <: Real}(::Type{Tuple{R, R}}, x::AbstractString)
  try 
    a = parse(R, x)
    return (a, a)
  catch ArgumentError
    return ((parse(R, y) for y in split(x, '-'))...) :: Tuple{R, R}
  end
end

s = ArgParseSettings()

@add_arg_table s begin
  "train"
    help = "train a model to fit synthesized images that imitate microscopy data"
    action = :command
  "test"
    help = "show an example image and test the model performance on it"
    action = :command
end

@add_arg_table s["train"] begin
  "--model", "-m"
    help = "model to be used - one of 'unetlike', 'multiscale3', fcrna'"
    arg_type = String
    default = "multiscale3"
  "--res", "-r"
    help = "resolution of the quadratic image patches used for training / testing"
    arg_type = Int
    default = 256
  "--sample-size", "-s"
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
  "--shuffle", "-p"
    help = "shuffle patches in the training set for each epoch"
    action = :store_true
  "--model-path", "-M"
    help = "save the model in this file after training"
    arg_type = String
    default = ""
  "--jitter", "-j"
    help = "randomness of the annotated positions in the label"
    arg_type = Int
    default = 0
  "--cell-number", "-n"
    help = "range of allowed cell-numbers per patch"
    arg_type = Tuple{Int, Int}
    default = (50, 80)
  "--cell-size", "-a"
    help = "range of the size of the two main axis of a cell in pixels"
    arg_type = Tuple{Float64, Float64}
    default = (3., 15.)
  "--cell-intensity", "-c"
    help = "range of the intensity a single cell may have"
    arg_type = Tuple{Float64, Float64}
    default = (0.2, 4.0)
  "--soften", "-S"
    help = "filtersize of the softening post-processing on the patches"
    arg_type = Int
    default = 2
  "--noise", "-N"
    help = "magnitude of the applied random noise on the patches"
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
end

@add_arg_table s["test"] begin
  "model"
    help = "path to the model that is to be used for testing"
  "--res", "-r"
    help = "resolution of the image patches to be tested"
    arg_type = Int
    default = 256
  "--cell-number", "-n"
    help = "range of allowed cell-numbers per patch"
    arg_type = Tuple{Int, Int}
    default = (50, 80)
  "--cell-size", "-a"
    help = "range of the size of the two main axis of a cell in pixels"
    arg_type = Tuple{Float64, Float64}
    default = (3., 15.)
  "--cell-intensity", "-c"
    help = "range of the intensity a single cell may have"
    arg_type = Tuple{Float64, Float64}
    default = (0.2, 4.0)
  "--soften", "-S"
    help = "filtersize of the softening post-processing on the patches"
    arg_type = Int
    default = 2
  "--noise", "-N"
    help = "magnitude of the applied random noise on the patches"
    arg_type = Float64
    default = 0.2
end

args = parse_args(s)


# ---------------------------------------------------------------------------- #
# Parsing done. Now load DCellC and execute the program

using Knet
include("../src/DCellC.jl")
using DCellC

if args["%COMMAND%"] == "train"

  args = args["train"]

  if args["model"] == "unetlike"
    model = UNetLike(GreyscaleImage, bn=args["batch-normalization"])
  elseif args["model"] == "multiscale3"
    model = Multiscale3(GreyscaleImage, bn=args["batch-normalization"])
  elseif args["model"] == "fcrna"
    model = FCRNA(GreyscaleImage, bn=args["batch-normalization"])
  else
    println(STDERR, "Invalid argument for option '--model' given. See --help")
  end

  res = args["res"]
  gen() = synthesize(args["res"], args["res"], args["cell-number"],
                     jitter = args["jitter"],
                     cell = SharpEllipticCell(args["cell-size"], args["cell-intensity"]),
                     pp = [PP.soften(args["soften"]), PP.noise(args["noise"])])



  train_data = [ gen() for i in 1:args["sample-size"] ]
  test_data = [ gen() for i in 1:args["test-size"] ]

  path = args["model-path"]
  if path == ""
    path = nothing
  end

  if args["optimizer"] == "adam"
    opt = Adam
  elseif args["optimizer"] == "rmsprop"
    opt = Rmsprop
  elseif args["optimizer"] == "nesterov"
    opt = Nesterov
  else
    println(STDERR, "Optimizer $(args["optimizer"]) not recognized")
  end

  if args["no-gpu"] || gpu() < 0
    println("# CPU Mode")
    at = Array{Float32}
  else
    println("# GPU Mode ($(gpu()))")
    at = KnetArray{Float32}
  end

  train!(model, train_data, 
         epochs = args["epochs"], 
         lr = args["learning-rate"],
         modelpath = path,
         batchsize = args["batch-size"],
         shuffle = args["shuffle"],
         testset = test_data,
         record = false,
         opt = opt,
         kernelsize = args["proximity-kernel-size"],
         peakheight = args["proximity-kernel-height"],
         at = at)


elseif args["%COMMAND%"] == "test"
  import ImageView
  using Gtk.ShortNames

  args = args["test"]

  res = args["res"]
  gen() = synthesize(args["res"], args["res"], args["cell-number"],
                     cell = SharpEllipticCell(args["cell-size"], args["cell-intensity"]),
                     pp = [PP.soften(args["soften"]), PP.noise(args["noise"])])

  img, lbl = gen()

  lbls = [ lbl ]

  if args["model"] != nothing
    @printf "load model from %s...\n" args["model"]
    model = load(args["model"])
    dens  = densitymap(model, img)
    dlbl, cds = label(dens, candidates = true)
    adj  = adjacency(dlbl, lbl)

    @printf "calculate metrics...\n" 
    @printf "loss:    %.3f\n" loss(model, img, lbl)
    @printf "count:   %d / %d (%.3f)\n" length(dlbl) length(lbl) (length(dlbl) / length(lbl)) 
    @printf "meanadj: %.3f\n" mean(adj)
    @printf "maxadj:  %.3f\n" maximum(adj)

    @printf "display test image\n"

    push!(lbls, dlbl)
    push!(lbls, cds)
    guidict = imshow(img, lbls)
    imshow(GreyscaleImage(proximitymap(args["res"], args["res"], lbl)))
    imshow(GreyscaleImage(dens))

  else
    @printf "no model provided - only display test image\n"
    guidict = imshow(img, lbls)
  end
  

  if (!isinteractive())
    c = Condition()
    win = guidict["gui"]["window"]
    signal_connect(win, :destroy) do widget
        notify(c)
    end
    wait(c)
  end

end

