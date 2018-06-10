

function autoscale{R <: Real}( img :: Array{R} )
  m = min(minimum(img), 0)
  d = max(maximum(img), 1) - m
  return clamp.((img - m) / d, 0, 1)
end

second{T, S}(t :: Tuple{T, S}) = t[2]


# --------------------------------------------------------------------------- #
# Import external functionality

#import Colors
#import ImageView

# --------------------------------------------------------------------------- #
# Utility function that shows a (greyscale) image as well as labels

function imshow( img :: GreyscaleImage, label :: Label = Label(zeros(2,0)); 
                 scale = 1., radius = 2, color=(0.1, 0.7, 0.2) )

  gdict = ImageView.imshow(scale * img.data)
  for (x,y) in label
    ImageView.annotate!(gdict, ImageView.AnnotationPoint(x, y, 
                                                         shape='.', 
                                                         size=radius, 
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
    for i in 1:length(label[j])
      ImageView.annotate!(gdict, ImageView.AnnotationPoint(label[j][i]..., 
                                                           shape='.', size=radius, 
                                                           color=Colors.RGB(color[j]...)))
    end
  end
  return gdict
end


# --------------------------------------------------------------------------- #
# Merging algorithms for labels

# adapted from 
# https://stackoverflow.com/questions/19375675/simple-way-of-fusing-a-few-close-points
function declutter(lbl :: Label, dist :: Real)
  if dist <= 0
    return lbl
  end

  data = Tuple{Float64, Float64}[]
  n = length(lbl)
  merged = zeros(Bool, n)
  for i in 1:n
    if !merged[i]
      count = 1
      x, y = lbl[i]
      for j in (i+1):n
        v, w = lbl[j]
        if !merged[j] && (x-v)^2 + (y-w)^2 <= dist^2
          x += v
          y += w
          count += 1
          merged[j] = true
        end
      end
      merged[i] = true
      push!(data, (x / count, y / count))
    end
  end
  return Label(data)
end


function merge(lbl :: Label, lbl2 :: Label, dist :: Real)
  if dist <= 0
    return join(lbl, lbl2)
  end

  data = copy(lbl2.data)
  for cm in lbl
    filter!(data) do ca
      norm([(cm .- ca)...]) > dist
    end
  end
  return join(lbl, Label(data))
end


