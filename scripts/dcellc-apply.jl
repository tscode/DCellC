
function args_apply(s)
  @add_arg_table s begin
    "--label", "-l"
      help = "should label files be generated?" 
      action = :store_true
    "--density", "-d"
      help = "should density images be generated?"
      action = :store_true
    "--no-gpu", "-g"
      help = "prevent usage of gpu even if support is detected"
      action = :store_true
    "--quiet", "-q"
      help = "do not print output"
      action = :store_true
    "--acceptance-level", "-a"
      help = "level used to determine which density peaks are selected for annotation"
      arg_type = Float64
      default = 50.
    "--assessment-level", "-A"
      help = "additional density level for which counts are given; only affects logging"
      arg_type = Float64
      default = 30.
    "--patchsize", "-p"
      help = "maximal patchsize to use for applying the network"
      arg_type = Int
      default = 256
    "--overlap"
      help = "overlap used for dividing the image in patches"
      arg_type = Int
      default = 24
    "model"
      required = true
      arg_type = String
      help = "DCellC model (.dccm file) to be used"
    "images"
      nargs='+'
      required = true
      arg_type = String
      help = "image files that the model is applied to"
  end
end


# first command can be "synth" or "data"
function cmd_apply(args)

  # load the model
  model = modelload(args["model"])

  # check if gpu support should be disabled
  at = args["no-gpu"] ? Array{Float32} : nothing

  # iterate through all images
  for imgfile in args["images"]

    img = imgload(imgfile)

    # calculate density maps
    dens = density_patched(model, img, 
                           patchsize = args["patchsize"], 
                           overlap = args["overlap"], at = at) 

    # calculate labels and candidate levels
    aclevel = args["acceptance-level"]
    aslevel = args["assessment-level"]
    lbl  = label(dens, level = aclevel)
    clbl = label(dens, level = aslevel)

    # print out classification information
    if !args["quiet"]
      @printf("Image %s: %d (%.0f) %d (%.0f)", imgfile, 
              length(lbl), aclevel, length(clbl), aslevel)
    end

    # write a label file if so desired
    if args["label"]
      lblsave(splitext(imgfile)[1], lbl)
    end

    # write a density image if so desired
    if args["density"]
      lblsave(splitext(imgfile)[1]*"-dens.tif", GreyscaleImage(dens))
    end
  end
end
