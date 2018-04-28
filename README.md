
# DCellC -- Deep Cell Counting 

This julia library provides functionality regarding cell counting in
microscopy images. It relies on `Knet.jl` to train fully convolutional
neural networks to recognize and count cell structures.

Note that this project is currently in early stages of development and
might be unusable -- depending on the mood of the author.

## Installation
In order to use `DCellC.jl`, you need a working version of `julia 0.6`.
Further dependencies are
```
Knet.jl
Colors.jl
ImageFiltering.jl
JLD2.jl.
```
They should be fetched automatically when installing this package via
```julia
Pkg.clone("...")
```
from a `julia 0.6` REPL. 

## Usage

Good Question

## What is it and how does it work?
Counting the number of cells in microscopy images of all kinds (migration
assays, ...) is a pervasive problem in the quantitative study of biological
phenomena. Since the process of manual counting is vexing -- and
pulverizing countless hours of potentially productive work for students and
assistants in the live sciences all over the world -- a number of tools
have been developed to automate this process and make it more reliable. 

Classical image processing approaches to this task, like
[CellProfiler](http://cellprofiler.org/),
[CELLCOUNTER](https://www.hindawi.com/journals/bmri/2014/863564/),
or the [Imagej](https://imagej.net) plugin
[Cell-Counter](https://imagej.nih.gov/ij/plugins/cell-counter.html)
have been developed and achive remarkable results for certain tasks of
microscopic image analysis. Drawbacks of these tools are the manual fine
tuning that is generally required in order for them to work fine.

In the recent years, with the advent of machine learning techniques for
various image processing applications, different methods for identifying
and counting cells, based on neural networks, have been proposed. This
project, `DCellC.jl`, gathers much of its ideas from the publications 
[(Xie, Noble, Zisserman; 2015)](https://www.tandfonline.com/doi/abs/10.1080/21681163.2016.1149104) and
[(Pan, Yang, Li, et al.; 2018)](https://link.springer.com/article/10.1007%2Fs11280-017-0520-7). 
The basic thought is that fully convolutional neural networks should be
able to filter images for instances of similar cell-shaped realizations.
The network is taught to produce "proximity maps" (an image of black dots
wherever the network thinks a cell is lurking) out of the original
images by means of convolutions with trainable convolution kernels. This
reduces finding cells to a classical regression problem: Minimize the
distance between the proximity maps and the true labels by adapting the
network's weights.

Depending on the quality of the training data (labeled microscopy images
with center positions for the cells) these methods can be used to
recover a majority of the cells with little parameter tuning at all (or so
I hope). Furthermore, it should (theoretically) work on various kinds of
microscopic images, and have a strong robustness to focus on relevant
information, such that pre or post-processing steps should be unnecessary.

## Todo

* Everything
