#! /usr/bin/env julia

println("Start test 'tests/loadmodel.jl'")

include("../src/DCellC.jl")
using DCellC

model = UNetLike(GreyscaleImage, bn = true)

println("Test loadmodel.jl: Created model successfully")

mname = tempname()
modelsave(mname, model)

println("Test loadmodel.jl: Saved model successfully")

model2 = modelload(mname)

@assert weights(model) == weights(model2)
#@assert state(model) == state(model2)

println("Test loadmodel.jl: Loaded model sucessfully")
println("Test loadmodel.jl completed")
