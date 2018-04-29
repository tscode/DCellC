
# --------------------------------------------------------------------------- #
# Circumvent current limitations of Knet for cat(3, ...) with 4-dim arrays


# Copied from issue #198 of Knet.jl
function setidx!(a::KnetArray, v, I...) 
    crange = CartesianRange(to_indices(a, I))
    linind = [sub2ind(size(a), t.I...) for t in crange]
    setindex!(a, v, vec(linind))
end

function cat3(a :: KnetArray{Float32, 4}, b :: KnetArray{Float32, 4})
  sa, sb = size(a), size(b)

  @assert sa[[1,2,4]] == sb[[1,2,4]]

  c = similar(a, sa[1], sa[2], (sa[3]+sb[3]), sa[4])
  setidx!(c, a, :, :, 1:sa[3], :)
  setidx!(c, b, :, :, (sa[3]+1):(sa[3]+sb[3]), :)
  return c
end


function cat3(a :: Array{Float32, 4}, b :: Array{Float32, 4}) 

  sa, sb = size(a), size(b)

  @assert sa[[1,2,4]] == sb[[1,2,4]]

  _a = reshape(a, prod(sa[1:2]), sa[3:4]...)
  _b = reshape(b, prod(sb[1:2]), sb[3:4]...)

  return reshape(hcat(_a, _b), sa[1:2]..., sa[3]+sb[3], sa[4])
end


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

