
# --------------------------------------------------------------------------- #
# Circumvent current limitations of Knet for cat(3, ...) with 4-dim arrays


# Copied from issue #198 of Knet.jl
function setidx!(a::KnetArray, v, I...) 
    crange = CartesianRange(to_indices(a, I))
    linind = [sub2ind(size(a), t.I...) for t in crange]
    setindex!(a, v, vec(linind))
end

# This can only be done efficiently with knet-Arrays in case of 
# batchsize = 1, when permutedims is a no-op
function cat3(a, b)

  if size(a, 4) == 1
    k = size(a)[1:2]
    sa, sb = size(a, 3), size(b, 3)

    ar, br = reshape(a, prod(k), sa), reshape(b, prod(k), sb)
    cr = reshape(hcat(ar, br), k..., sa + sb, 1)

    return cr

  else
    ap, bp = permutedims(a, (1,2,4,3)), permutedims(b, (1,2,4,3))
    sa, sb = size(ap, 4), size(bp, 4)

    k = size(ap)[1:3]
    ap, bp = reshape(ap, prod(k), sa), reshape(bp, prod(k), sb)

    cp = hcat(ap, bp)
    cp = reshape(cp, k..., sa + sb)
  
    return permutedims(cp, (1, 2, 4, 3))
  end

end


cat3(a :: Array{Float32, 4}, b :: Array{Float32, 4}) = cat(3, a, b)


# --------------------------------------------------------------------------- #
# Primitives for multires networks

# Padding for convolution such that dimensions are preserved
pad(w) = floor(Int, size(w, 1) / 2)

# Convolution and relu activation 
rconv(w, x) = relu.(conv4(w, x, padding=pad(w))) 

# Convolution, batch normalization, and relu activation
rbconv(wb, m, wc, x) = relu.(batchnorm(conv4(wc, x, padding=pad(wc)), m, wb))

# Up-convolution
uconv(w, x) = deconv4(w, x, stride=2, padding=1)

# Up-convolution and batch normalization
ubconv(wb, m, wc, x) = batchnorm(deconv4(wc, x, stride=2, padding=1), m, wb)

# Up-convolution and stacking
uconv(w, x, y) = cat3(x, deconv4(w, y, stride=2, padding=1))

# Up-convolution, batch normalization, and stacking
ubconv(wb, m, wc, x, y) = 
    cat3(x, batchnorm(deconv4(wc, y, stride=2, padding=1), m, wb))


# Default initialization for convolution kernel
wr(i, o, size = (3, 3)) = gaussian(Float32, size..., i, o, std=0.060)

# Default initialization for up-convolution kernel
wu(i, o) = bilinear(Float32, 2, 2, o, i)

# Knet parameters for batch normalization
bn(c) = bnparams(Float32, c)


# Check if a number is odd 
odd(x::Integer) = (x % 2 != 0)

