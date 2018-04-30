
# --------------------------------------------------------------------------- #
# Prepare layout for batches of images, weights, states and prmaps
# For suitable arraytypes, this will transfer the data into the graphics
# memory

function packimage(img :: Image; at = Array{Float32})
 pack = reshape(imgdata(img), imgsize(img)..., imgchannels(img), 1)
 return convert(at, pack)
end

function packimage(imgs :: Vector{<: Image}; at = Array{Float32})
  pack = cat(4, map(packimage, imgs)...)
  return convert(at, pack)
end

function packweight(w :: Vector{Any}; at = Array{Float32})
  return map(x -> convert(at, x), w)
end

function packstate(s :: Vector{Any}; at = Array{Float32})
  return s
  # Cannot convert Knet Moments to gpu values!
  #return map(x -> convert(at, x), s)
end

function packproxymap(prmap; at = Array{Float32})
  if length(size(prmap)) < 3
    return convert(at, reshape(prmap, size(prmap)..., 1))
  else
    return convert(at, prmap)
  end
end

# --------------------------------------------------------------------------- #
# Check consistency between Array types

# TODO: This should be possible without relying on explicit 
# knowledge of KnetArrays
arraytype(a :: Array{Float32}) = Array{Float32}
arraytype(a :: KnetArray{Float32}) = KnetArray{Float32}


# --------------------------------------------------------------------------- #
# Loss function 


function loss(w, s, imgdata, prmap, 
              mt :: Type{<: Model}; at = nothing, kwargs...)
  
  # Loss that can be applied to raw data without conversion 
  # if `at` is not specified. This can be used to efficiently 
  # calculate the loss for already gpu-packed data. If `at` 
  # is specified, conversion might take place.

  if at != nothing
    w       = packweight(w, at = at)
    s       = packstate(s,  at = at)
    prmap   = packproxymap(prmap, at = at)
    imgdata = packimage(imgdata, at = at)
  end
  
  return sum(abs2, density(w, s, imgdata, mt) .- prmap) / length(prmap)
end


function loss(w, s, imgdata, lbls :: Vector{Label}, 
              mt :: Type{<: Model}; at = nothing, kwargs...)

  # This function will build proximity maps from the given labels.
  # If the argument `at` is not specified, these proximity maps
  # inherit their arraytype from `imgdata`

  _at = (at != nothing) ? at : arraytype(imgdata)

  imgdata = packimage(imgdata, at = _at)
  prmap   = proxymap(imgsize(imgdata)..., lbls; kwargs...)
  prmap   = packproxymap(prmap)
  return loss(w, s, imgdata, prmap, mt; at = at, kwargs...)
end


function loss{I <: Image}(w, s, img :: I, lbl :: Label, 
                          mt :: Type{<: Model{I}}; at = nothing, kwargs...)

  # Loss calculated for an image
  # Creates proximity maps and then calculates the loss

  _at = (at != nothing) ? at : arraytype(imgdata(img))

  x = packimage(img, at = _at)
  return loss(w, s, x, [lbl], mt; at = at, kwargs...)
end

function loss(m :: Model, img :: GreyscaleImage, lbl :: Label) 
  return loss(weights(m), state(m), img, lbl, typeof(m))
end


# --------------------------------------------------------------------------- #
# Loss gradient 

_gradloss = gradloss(loss)  # Calculate both the gradient and loss value
_grad     = grad(loss)

lossgrad(args...; kwargs...) = _gradloss(args...; kwargs...)[2:-1:1]

function lossgrad{I <: Image}(m :: Model{I}, img :: I, lbl :: Label; kwargs...) 
  return lossgrad(weights(m), state(m), img, lbl, typeof(m); kwargs...)
end


# --------------------------------------------------------------------------- #
# Divide dataset in batches

function makebatches(imgs, lbls, batchsize; shuffle = false)
  n = length(imgs)
  idx = shuffle ? Base.shuffle(collect(1:n)) : collect(1:n)
  sel = collect(1:batchsize:n)
  m = length(sel)

  idxs = [ idx[sel[i] : (i < m ? sel[i+1] - 1 : n)] for i in 1:m ]
  return [ (imgs[range], lbls[range]) for range in idxs ]
end

function packbatches(batches; at = Array{Float32}, kwargs...)
  return map(batches) do batch
    imgs   = packimage(batch[1], at = at)
    prmaps = proxymap(imgsize(imgs)..., batch[2]; kwargs...)
    prmaps = packproxymap(prmaps, at = at)
    return (imgs, prmaps)
  end
end


# --------------------------------------------------------------------------- #
# High level training function

"""
    train!(model, imgs, lbls; kwargs...)

    train!(model, lmgs; kwargs...)

Train a model `model` using the images `imgs` and labels `lbls`. Alternatively, an iterable `lmgs` of labeled images can be provided.

# Arguments
- `testset`: an iterable of labeled images to be used for testing the network.
- `epochs = 10`: the number of epochs (iterations of the training images)
  used for training.
- `batchsize = 1`: the size of a batch used for one update step of the model.
- `shuffle = true`: whether to shuffle the training images for each epoche
- `log = true`: whether to print information about the model's training
  performance.
- `logpath`: redirect logging to this file.
- `modelpath`: file to store the trained model in
- `record = true`: record the training performance in a `Dict` and return it.
- `patchsize`: providing an integer for this option causes that the
  effective training images are patches of this size, generated from the
  input image during each epoche.
- `patchmode`: if `patchmode < 1`, create patches by cutting ordered
  patches from the given input images. If `patchmode >= 0`, create
  `patchmode` patches by random selection.
- `at = Array{Float32}`: the arraytyped used for training and evaluating.
  Currently, this is restricted to `Array{Float32}` for cpu computation or
  `KnetArray{Float32}` for gpu computation.
- `kernelsize = 7`: size of the kernel used to construct proximity maps.
- `peakheight = 100`: height of the kernels used to construct proximity
  maps.
- `opt = Adam`: Knet optimizer to be used. All other keyword arguments are
  passed to the constructor of the optimizer.
"""

function train!(model :: Model, imgs, lbls; 
                epochs = 10, 
                opt = Adam, 
                log = true, 
                logpath = nothing, 
                record = true, 
                batchsize = 1, 
                patchsize = nothing, 
                patchmode = 0,
                shuffle = false, 
                modelpath = nothing, 
                testset = nothing, 
                kernelsize = 7, 
                peakheight = 100., 
                at = nothing, 
                kwargs...)

  # Sanity checks

  @assert (length(imgs) == length(lbls))
  @assert all(x -> x == imgsize(imgs[1]), imgsize.(imgs))

  # Pack the model weights and state.
  # In practice this means to transfer them to 
  # gpu memory if a gpu based arratype is given.

  if at == nothing
    at = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}
  end

  w = packweight(weights(model), at = at)
  s = packstate(state(model), at = at)

  # Initialize the optimizers and conduct some 
  # general preparations

  optim = optimizers(w, opt; kwargs...)
  test  = (testset != nothing) && (log || record)

  if patchsize == nothing
    n = ceil(Int, length(imgs) / batchsize)
  elseif patchmode > 0
    n = ceil(Int, length(imgs) * patchmode / batchsize)
  else
    # WARNING: This will not work if margins or manual offsets are given to
    # `ordered_patches`
    m1 = floor(imgsize(imgs[1])[1] / patchsize)
    m2 = floor(imgsize(imgs[1])[2] / patchsize)
    n = ceil(Int, length(imgs) * m1 * m2 / batchsize)
  end

  if record
    rec = Dict{Symbol, Array{Float64}}( 
               :loss_train    => zeros(n, epochs),
               :count_train   => zeros(n, epochs),
               :meanadj_train => zeros(n, epochs),
               :maxadj_train  => zeros(n, epochs))
    if test
      rec[:loss_test]    = zeros(epochs)
      rec[:count_test]   = zeros(epochs)
      rec[:meanadj_test] = zeros(epochs)
      rec[:maxadj_test]  = zeros(epochs)
    end
  end

  # Bring the testset in the correct format

  if test
    if typeof(testset) <: NTuple{2}
      timgs, tlbls = testset
    else
      timgs = [ t[1] for t in testset ]
      tlbls = [ t[2] for t in testset ]
    end
    tbatches = makebatches(timgs, tlbls, batchsize, shuffle = false)
    tpacks   = packbatches(tbatches, at = at,
                           peakheight = peakheight,
                           kernelsize = kernelsize)
  end

  # Print overview

  if log
    @printf "# CellCC training progress\n"
    @printf "#\n"
    @printf "# model      %s\n" typeof(model)
    @printf "# date       %s\n" now()
    @printf "# images     %d\n" length(imgs)
    @printf "# imagesize  %s\n" imgsize(imgs[1])
    @printf "# patchsize  %s\n" patchsize
    @printf "# patchmode  %s\n" patchmode
    @printf "# batches    %d\n" n
    @printf "# batchsize  %d\n" batchsize
    @printf "# epochs     %d\n" epochs
    @printf "# testset    %s\n" test ? length(timgs) : "-"
    @printf "# shuffle    %s\n" shuffle
    @printf "# record     %s\n" record
    @printf "# optimizer  %s\n" optim[1]
    @printf "# modelpath  %s\n" modelpath == nothing ? "-" : modelpath
    @printf "# logpath    %s\n" logpath == nothing ? "-" : logpath
  end

  # Start the training loop

  time = @elapsed for i in 1:epochs

    # Generate batches used for training. These batches 
    # are also 'packed', meaning that they are brought in 
    # a dense format consisting of image and proximity 
    # map data arrays. Depending on `at`, this might 
    # transfer the batch-data to gpu memory.

    # TODO: make the batch-acquicision process more dynamic!
    # * Allow for pipelines

    if patchsize != nothing && patchmode > 0
      rp = random_patches.(imgs, lbls, patchmode, 
                           size = (patchsize, patchsize))
      timgs, tlbls = vcat(first.(rp)...), vcat(second.(rp)...)

    elseif patchsize != nothing && patchmode <= 0
      op = ordered_patches.(imgs, lbls,
                            size = (patchsize, patchsize))
      timgs, tlbls = vcat(first.(op)...), vcat(second.(op)...)

    else
      timgs, tlbls = imgs, lbls
    end

    batches = makebatches(timgs, tlbls, batchsize, shuffle = shuffle)
    packs   = packbatches(batches, at = at, 
                          peakheight = peakheight, 
                          kernelsize = kernelsize) 

    if log
      @printf "\n"
      @printf "# Epoche  Batch   Loss  Count  MeanAdj   MaxAdj \n"
      @printf "# ----------------------------------------------\n"
    end

    imgsave("image", imgs[1])
    imgsave("label", GreyscaleImage(proxymap(imgsize(imgs[1])..., lbls[1])))

    # Training epoche

    for j in 1:n

      # Make sure that everything is printed

      flush(STDOUT)
      flush(STDERR)

      # Extract image data and proximity maps
 
      data, prmaps = packs[j]
      lbl = batches[j][2]

      imgsave("image$j", batches[j][1][1])
      imgsave("label$j", GreyscaleImage(proxymap(imgsize(batches[j][1][1])..., 
                                                batches[j][2][1])))

      l, lg = lossgrad(w, s, data, prmaps, typeof(model))
      update!(w, lg, optim)

      if log || record

        # Let the model predict labels

        dens = density(w, s, data, typeof(model))
        @show maximum(dens)
        @show minimum(dens)
        dlbl = label(w, s, data, typeof(model))

        # Evaluate the quality of the prediction

        c      = mean(length.(dlbl) ./ length.(lbl))
        adj    = adjacency.(dlbl, lbl)
        mm, mx = mean(mean, adj), mean(maximum, adj)

        if log
          @printf("  %6d  %5d  %5.1f  %5.3f  %7.2f  %7.2f\n", 
                  i, j, l, c, mm, mx)
        end

        if record
          rec[:loss_train][j, i]    = l
          rec[:count_train][j, i]   = c
          rec[:meanadj_train][j, i] = mm
          rec[:maxadj_train][j, i]  = mx
        end
      end
    end

    # Evaluate performance on the test set

    if test

      l = c = mm = mx = 0.

      for j in 1:length(tpacks)
        data, prmaps = tpacks[j]
        timg, tlbl = tbatches[j]

        dlbl = label(w, s, data, typeof(model))
        adj  = adjacency.(dlbl, tlbl)

        l  += loss(w, s, data, prmaps, typeof(model)) * length(timg)
        c  += sum(length.(dlbl) ./ length.(tlbl))
        mm += sum(mean, adj)
        mx += sum(maximum, adj)
      end

      l  /= length(timgs)
      c  /= length(timgs)
      mm /= length(timgs)
      mx /= length(timgs)

      if log
          @printf "\n"
          @printf("  %6d      0  %5.1f  %5.3f  %7.2f  %7.2f\n", 
                   i, l, c, mm, mx)
      end

      if record
        rec[:loss_test][i]    = l 
        rec[:count_test][i]   = c
        rec[:meanadj_test][i] = mm
        rec[:maxadj_test][i]  = mx
      end
    end
  end

  # Update the model weights and its state

  weights(model)[:] = convert.(Array{Float32}, w)
  state(model)[:]   = s

  @printf "\n"
  @printf "# End of training process after %.1f seconds\n" time

  if logpath != nothing
    # TODO
    @printf "# Warning: Logpath not yet implemented\n"
    @printf "# Did not save a version of this output in file %s\n" logpath
  end

  if modelpath != nothing
    save(model, modelpath)
    open(modelpath * ".sd", "w") do file serialize(file, model) end
    @printf "# Model saved at %s\n" modelpath
  end

  return record ? rec : nothing
end


function train!(model :: Model, data; kwargs...)
  imgs = [ d[1] for d in data ]
  lbls = [ d[2] for d in data ]
  train!(model, imgs, lbls; kwargs...)
end

function train!(model :: Model, gen :: Function; samples = 10, kwargs...)
  data = [gen() for i in 1:samples]
  train!(model, data; kwargs...)
end


# --------------------------------------------------------------------------- #
# Function for training with a single image frame
# Intended for testing purposes

function train!(model :: Model, img :: Image, lbl :: Label; 
                epochs = 10, record = true, opt = Adam, 
                log = true, logpath = nothing, modelpath = nothing, kwargs...)

  optim = optimizers(weights(model), opt; kwargs...)

  if record
    rec = Dict{Symbol, Vector{Float64}}( 
               :loss    => zeros(epochs),
               :count   => zeros(epochs),
               :meanadj => zeros(epochs),
               :maxadj  => zeros(epochs))
  end

  if log
    @printf "# CellCC training progress (single image mode)\n"
    @printf "#\n"
    @printf "# model      %s\n" typeof(model)
    @printf "# date       %s\n" now()
    @printf "# epochs     %d\n" epochs
    @printf "# record     %s\n" record
    @printf "# optimizer  %s\n" optim[1]
    @printf "# modelpath  %s\n" modelpath == nothing ? "-" : modelpath
    @printf "# logpath    %s\n" logpath == nothing ? "-" : logpath
    @printf "\n"
    @printf "# Epoche   Loss  Count  MeanAdj   MaxAdj\n"
    @printf "# --------------------------------------\n"
  end

  time = @elapsed for i in 1:epochs
    lg, l = lossgrad(model, img, lbl)
    update!(weights(model), lg, optim)

    if log || record

      dens = density(model, img)
      dlbl = label(dens)

      c      = length.(dlbl) ./ length.(lbl)
      adj    = adjacency(dlbl, lbl)
      mm, mx = mean(adj), maximum(adj)

      if log
        @printf("  %6d  %5.1f  %5.3f  %7.2f  %7.2f\n", 
                 i, l, c, mm, mx)
      end

      if record
        rec[:loss][i]    = l
        rec[:count][i]   = c
        rec[:meanadj][i] = mm
        rec[:maxadj][i]  = mx
      end
    end
  end

  @printf "\n"
  @printf "# End of training process after %.1f seconds\n" time

  if logpath != nothing
    # TODO
    @printf "# Warning: Logpath not yet implemented\n"
    @printf "# Did not save a version of this output in file %s\n" logpath
  end

  if modelpath != nothing
    save(model, modelpath)
    @printf "# Model saved at %s\n" modelpath
  end

  return record ? rec : nothing
end

