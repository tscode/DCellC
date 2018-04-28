
# --------------------------------------------------------------------------- #
# Import external functionality

#=import Colors=#
#=import ImageView=#

# --------------------------------------------------------------------------- #
# Utility function that shows a (greyscale) image as well as labels

function imshow( img :: GreyscaleImage, label :: Label = Label(zeros(2,0)); 
                 scale = 1., radius = 2, color=(0.1, 0.7, 0.2) )

  gdict = ImageView.imshow(scale * img.data)
  for i in 1:size(label, 2)
    ImageView.annotate!(gdict, ImageView.AnnotationPoint(label.data[:, i]..., 
                                                         shape='.', size=radius, 
                                                         color=Colors.RGB(color...)))
  end
  return gdict
end

function imshow( img :: GreyscaleImage, label; 
                 scale = 1., radius = 2, color = nothing )

  if color == nothing
    color = repeat([ (0.1, 0.7, 0.2), 
                     (0.1, 0.2, 0.7), 
                     (0.7, 0.2, 0.1) ], outer=ceil(Int, length(label)/3))
  end

  gdict = ImageView.imshow(scale * img.data)
  for j in 1:length(label)
    for i in 1:size(label[j], 2)
      ImageView.annotate!(gdict, ImageView.AnnotationPoint(label[j].data[:, i]..., 
                                                           shape='.', size=radius, 
                                                           color=Colors.RGB(color[j]...)))
    end
  end
  return gdict
end

