#! /usr/bin/env julia

println("Start test 'tests/lesson.jl'")

include("../src/DCellC.jl")
using DCellC

lmgs = [ synthesize(512, 512, (300, 500)) for i in 1:3 ]

sel() = (rand(1:50), rand(1:50), rand(260:400), rand(260:400))
selections = [(lmg..., sel()) for lmg in lmgs] 

lesson1 = Lesson(FCRNA, 
                 imgtype = GreyscaleImage,
                 selections = selections,
                 patchmode = 2,
                 epochs = 2,
                 patchsize = 128)

lesson2 = Lesson(Multiscale3(GreyscaleImage, bn = true),
                 patchmode = 1,
                 epochs = 2,
                 patchsize = 64)

println("Test lesson.jl: Creating instance of Lesson works")

lname = tempname()
lessonsave(lname, lesson1)
println("Test lesson.jl: Saving Lessons works")

lesson3 = lessonload(lname)
println("Test lesson.jl: Loading Lessons works")

model  = train(lesson1, log=false)
model2 = train(lesson2, log=false)

println("Test lesson.jl: Learned the lesson")

println("Test lesson.jl completed")


