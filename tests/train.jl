#! /usr/bin/env julia

println("Start test 'tests/train.jl'")

include("../src/DCellC.jl")
using DCellC


model = Multiscale3(GreyscaleImage, bn = true, seed = 42)
println("Test train.jl: Creating instance of FCRNA works")

lmgs  = [ synthesize(256, 256, (1500, 2000)) for i in 1:2 ]
println("Test train.jl: Creating synthesized images works")

train!(model, lmgs, patchsize = 128, epochs = 1, patchmode = 2, lr = 1e-3)
train!(model, lmgs, epochs = 1, lr = 1e-3)
println("Test train.jl: Training of model was sucessful")

println("Test train.jl completed")
