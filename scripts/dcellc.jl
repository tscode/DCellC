#! /usr/bin/env julia

using ArgParse

function args_main(s)
  @add_arg_table s begin
    "apply"
    help = "apply a provided model on images"
    action = :command
    "train"
    help = "train a model with provided images with annotations"
    action = :command
    "lesson"
    help = "train a model according to a given lesson file (.dcct)"
    action = :command
    "test"
    help = "test the performance of a provided model on images with annotations"
    action = :command
    "synth"
    help = "use DCellC on data constructed by a synthetic module"
    action = :command
  end
end

include("dcellc-apply.jl")
include("dcellc-train.jl")
include("dcellc-lesson.jl")

function cmd_main(args)
  if args["%COMMAND%"] == "apply"
    cmd_apply(args["apply"])
  elseif args["%COMMAND%"] == "train"
    cmd_train(args["train"])
  elseif args["%COMMAND%"] == "lesson"
    cmd_lesson(args["lesson"])
  elseif args["%COMMAND%"] == "test"
    cmd_test(args["test"])
  elseif args["%COMMAND%"] == "synth"
    cmd_synth(args["synth"])
  end
end

s = ArgParseSettings()

args_main(s)

args_apply(s["apply"])
args_train(s["train"])
args_lesson(s["lesson"])
#args_test(s)
#args_synth(s)


# apply
# train
# test
# synth
#   - train
#   - test
#   - generate

using Knet
include("../src/DCellC.jl")
using DCellC

cmd_main(parse_args(s))
