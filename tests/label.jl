#! /usr/bin/env julia

println("Start test 'tests/label.jl'")

include("../src/DCellC.jl")
using DCellC

label1 = Label()
label2 = Label([(1,2), (500, 33)])
label3 = Label(rand(1:500, 2, 20))

println("Test label.jl: Construction of labels works")

push!(label1, (1, 88))
deleteat!(label3, 18)
println("Test label.jl: Pushing and deleting in labels works")


for (x,y) in label2
  a = x*y
end
println("Test label.jl: Iterating over labels works")

lname = tempname()

lblsave(lname, label3)
label4 = lblload(lname)

@assert label4.data == label3.data
println("Test label.jl: Writing and loading labels works")

println("Test label.jl completed")
