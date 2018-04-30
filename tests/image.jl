#! /usr/bin/env julia

println("Start test 'tests/image.jl'")

include("../src/DCellC.jl")
using DCellC

gsimage = GreyscaleImage(rand(500, 600))
println("Test image.jl: Creating instance of GreyscaleImage works")

rgbimage = RGBImage(rand(500, 600, 3))
println("Test image.jl: Creating instance of RGBImage works")

ordered_patches(gsimage, 
                size = (64, 64),
                offset = nothing,
                margin = 2,
                shuffle = false,
                multitude = 2)

ordered_patches(rgbimage, 
                size = (64, 64),
                offset = nothing,
                margin = 2,
                shuffle = false,
                multitude = 2)

println("Test image.jl: Creating ordered patches works")


random_patches(gsimage, 20,
               size = (128, 128),
               multitude = 3)

random_patches(rgbimage, 20,
               size = (128, 128),
               multitude = 3)

println("Test image.jl: Creating random patches works")


@assert imgdata(gsimage)  == gsimage.data
@assert imgdata(rgbimage) == rgbimage.data

@assert imgsize(gsimage) == (500, 600)
@assert imgsize(rgbimage) == (500, 600)

@assert imgchannels(gsimage) == imgchannels(GreyscaleImage) == 1
@assert imgchannels(rgbimage) == imgchannels(RGBImage) == 3


println("Test image.jl: Auxiliary functions on images work")


gsname  = tempname() * ".tif"
rgbname = tempname() * ".tif"

imgsave(gsname, gsimage)
gsimage2 = imgload(gsname)

imgsave(rgbname, rgbimage)
rgbimage2 = imgload(rgbname)

@assert all(imgdata(gsimage2)  .- imgdata(gsimage)  .< 2./255)
@assert all(imgdata(rgbimage2) .- imgdata(rgbimage) .< 2./255)

println("Test image.jl: Saving and loading images works")


println("Test image.jl completed")
