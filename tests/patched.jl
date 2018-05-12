#! /usr/bin/env julia

println("Start test 'tests/patched.jl'")

include("../src/DCellC.jl")
using DCellC

image, label = synthesize(512, 512, (500, 1000))
model = Multiscale3(GreyscaleImage, bn = true, seed = 42)

dens = density(model, image)

println("Test patched.jl: Calculating density (unpatched) successful")

dens2 = density_patched(model, image, patchsize=200)

println("Test patched.jl: Calculating density (patched) successful")

error = mean(abs.(dens - dens2))

println("Test patched.jl: Mean difference between patched and unpatched density: $error")

println("Test patched.jl completed")
