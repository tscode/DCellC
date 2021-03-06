
# --------------------------------------------------------------------------- #
# IO functionality
# Saving and loading images, labels, models, ...

getext(fname) = splitext(fname)[2]
hasext(fname, ext) = getext(fname) == ext

joinext(fname, ext) = hasext(fname, ext) ? fname : fname * ext
joinext(fname, ext, autoext) = autoext ? joinext(fname, ext) : fname

# --------------------------------------------------------------------------- #
# Images

fileext(::Image) = ".tif"
fileext(::Type{<: Image}) = ".tif"

function imgsave(fname :: String, img :: GreyscaleImage, autoext = true)
  fname = joinext(fname, fileext(img), autoext)
  img = imgdata(img)
  FileIO.save(fname, convert(Array{Images.Gray{Images.N0f16}}, img))
end

function imgsave(fname :: String, img :: RGBImage, autoext = true)
  fname = joinext(fname, fileext(img), autoext)
  img = permutedims(imgdata(img), (3, 1, 2))
  FileIO.save(fname, Images.colorview(Images.RGB, img))
end

function imgload(fname :: String, autoext = true)
  fname = joinext(fname, fileext(Image), autoext)
  img = FileIO.load(fname)
  if eltype(img) <: Images.RGB
    img = permutedims(Images.channelview(img), (2, 3, 1))
    return RGBImage(img)
  else
    return GreyscaleImage(img)
  end
end


# --------------------------------------------------------------------------- #
# Labels 

fileext(::Label) = ".dccl"
fileext(::Type{Label}) = ".dccl"

function lblsave(fname :: String, lbl :: Label, autoext = true)
  fname = joinext(fname, fileext(lbl), autoext)
  writedlm(fname, convert(Matrix{Int}, lbl))
end

function lblload(fname :: String, autoext = true)
  fname = joinext(fname, fileext(Label), autoext)
  return Label(readdlm(fname, Int))
end


# --------------------------------------------------------------------------- #
# Labeled images

function lmgsave(fname :: String, limg :: LabeledImage, autoext = true)
  basename = splitext(fname)[1]
  if autoext
    imgsave(fname, limg[1])
    if hasext(fname, fileext(Image))
      lblsave(basename, limg[2])
    else
      lblsave(fname, limg[2])
    end
  else
    imgsave(fname, limg[1], false)
    lblsave(basename, limg[2], true)
  end
end

function lmgload(fname :: String, autoext = true)
  basename = splitext(fname)[1]
  if autoext
    if hasext(fname, fileext(Image))
      return imgload(fname), lblload(basename)
    else
      return imgload(fname), lblload(fname)
    end
  else
    return imgload(fname, false), lblload(basename, true)
  end
end

function lmgload(imgname :: String, lblname :: String, axtoext = true)
  if autoext
    return imgload(imgname), lblload(lblname)
  else
    return imgload(imgname, false), lblload(lblname, false)
  end
end

# --------------------------------------------------------------------------- #
# Models 

fileext(::Model) = ".dccm"
fileext(::Type{<: Model}) = ".dccm"

function modelsave(fname :: String, 
                   model :: Model, 
                   autoext = true; 
                   name :: String = "",
                   description :: String = "")

  fname = joinext(fname, fileext(model), autoext)
  name = (name == "") ? splitext(basename(fname))[1] : name
  # Do not use JLD2.@save, since this worked buggy on distributed 
  # file systems. See issue #55 of JLD2.jl. Disadvantage: Not using 
  # nmap probably is much slower, which should however not be a 
  # problem for the relatively small models used in this project 
  JLD2.jldopen(fname, true, true, true, IOStream) do file
    write(file, "model", model)
    write(file, "name", name)
    write(file, "description", description)
  end
end

function modelload(fname :: String,
                   autoext = true; 
                   name :: Bool = false, 
                   description :: Bool = false) 

  fname = joinext(fname, fileext(Model), autoext)
  returnname = name
  returndescr = description

  JLD2.@load fname model name description

  if returnname && returndescr return model, name, description
  elseif returndescr           return model, description
  elseif returnname            return model, name
  else                         return model
  end
end


# --------------------------------------------------------------------------- #
# Lessons 

fileext(::Lesson) = ".dcct"
fileext(::Type{<: Lesson}) = ".dcct"

function lessonsave(fname :: String, lesson :: Lesson, autoext = true)
  fname = joinext(fname, fileext(lesson), autoext)
  JLD2.jldopen(fname, true, true, true, IOStream) do file
    write(file, "lesson", lesson)
  end
end

function lessonload(fname :: String, autoext = true)
  fname = joinext(fname, fileext(Lesson), autoext)
  JLD2.@load fname lesson
  return lesson
end


