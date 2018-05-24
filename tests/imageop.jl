#! /usr/bin/env julia

println("Start test 'tests/imageop.jl'")

include("../src/DCellC.jl")
using DCellC

ops = [ Id(),
        Soften(3),
        FlipV(),
        FlipH(),
        Jitter(),
        PixelNoise(0.1),
        StretchV(1.1),
        StretchH(1.5)
      ]

println("Test imageop.jl: Creation of base image operations works")

ran = (Soften(3), 1) * (Soften(1), 2) * (Jitter(), 3)
pip = Soften(3) * (PixelNoise(0.1), 2) * ran

println("Test imageop.jl: Creation of pipelines and random operations works")

ops = [ops; [ran, pip]]

lmg = synthesize(256, 256, 1000)

for op in ops
  apply(op, lmg)
end

println("Test imageop.jl: Applying operations works")

println("Test imageop.jl completed")

