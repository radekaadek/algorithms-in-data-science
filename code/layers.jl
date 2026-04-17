include("autodiff.jl")

global IS_TRAINING = true

abstract type Operator end
const Chain = Vector{Operator}

struct Sigmoid <: Operator end
struct ReLU <: Operator end
struct MaxPool <: Operator end
struct Flatten <: Operator end
struct Dense <: Operator
  insize::Int64
  outsize::Int64
end
struct Dropout <: Operator
  p::Float64
end
struct SoftMax <: Operator end
struct Conv <: Operator
  kernel_size::Tuple{Int,Int}
  channels::Pair{Int,Int}
  pad::Int
  bias::Bool
end

relu() = ReLU()
sigmoid() = Sigmoid()
maxpool() = MaxPool()
flatten() = Flatten()
dropout(p::Float64) = Dropout(p)
softmax() = SoftMax()
Conv(kernel_size::Tuple{Int,Int}, channels::Pair{Int,Int}; pad=0, bias=false) = Conv(kernel_size, channels, pad, bias)
dense(pair::Pair{Int64,Int64}) = Dense(first(pair), last(pair))
dense(pair::Pair{Int64,Int64}, activation) = tuple(dense(pair), activation())

abstract type Loss end

struct BinaryCrossEntropy <: Loss end
bce(output, target) = BinaryCrossEntropy()(output, target)

struct CrossEntropy <: Loss end
cce(output, target) = CrossEntropy()(output, target)

struct Tensor{N}
  outsize::NTuple{N,Int64}
end
tensor(sz...) = Tensor(sz)

function chain(x::Tuple)
  y = Vector{Operator}()
  for v in x
    v isa Tuple ? append!(y, v) : push!(y, v)
  end
  return y
end

function (chain::Chain)(x)
  node = x
  for op in chain
    node = op(node)
  end
  return node
end

# --- Discretization & Primal/Adjoint Logic ---

(x::Tensor{N})() where N = GraphNode(zeros(x.outsize...))
(E::BinaryCrossEntropy)(x, y) = GraphNode(:bce, (x, y), zeros(1))
(y::Sigmoid)(x) = GraphNode(:sigmoid, (x,), zeros(length(x.data)))
(y::ReLU)(x) = GraphNode(:relu, (x,), zeros(length(x.data)))
(y::SoftMax)(x) = GraphNode(:softmax, (x,), zeros(length(x.data)))
(E::CrossEntropy)(x, y) = GraphNode(:cce, (x, y), zeros(1))

function (y::Dense)(x)
  # 1. Calculate the Glorot Uniform limit
  # limit = sqrt(6 / (fan_in + fan_out))
  limit = sqrt(6.0 / (y.insize + y.outsize))

  # 2. Initialize W with Glorot Uniform
  # rand(out, in) generates U(0, 1). 
  # We scale it to (0, 2*limit) and shift by -limit to get U(-limit, limit).
  W_init = (rand(y.outsize, y.insize) .* (2.0 * limit)) .- limit
  W = GraphNode(W_init, true)

  # 3. Initialize b to 0
  b = GraphNode(zeros(y.outsize), true)

  # 4. Connect to the graph
  mul = GraphNode(:mul, (W, x), zeros(y.outsize))
  return GraphNode(:add, (mul, b), zeros(y.outsize))
end

function (y::Dropout)(x)
  # Create helper nodes for the mask and probability
  # inside the graph, so that adjoint! can access them
  mask = GraphNode(zeros(size(x.data)))
  p_node = GraphNode([y.p])

  return GraphNode(:dropout, (x, mask, p_node), zeros(size(x.data)))
end

function (y::MaxPool)(x)
  W, H, C = size(x.data)
  out_W, out_H = div(W, 2), div(H, 2)
  return GraphNode(:maxpool2x2, (x,), zeros(out_W, out_H, C))
end

function (y::Flatten)(x)
  return GraphNode(:flatten, (x,), zeros(length(x.data)))
end

function (layer::Conv)(x)
  kW, kH = layer.kernel_size
  c_in, c_out = layer.channels

  # Inicjalizacja He (zgodnie z FAQ dla warstwy Conv)
  fan_in = kW * kH * c_in
  limit = sqrt(2.0 / fan_in)
  W_init = randn(kW, kH, c_in, c_out) .* limit
  W = GraphNode(W_init, true)

  pad_node = GraphNode([layer.pad])

  W_img, H_img, _ = size(x.data)
  out_W = W_img + 2 * layer.pad - kW + 1
  out_H = H_img + 2 * layer.pad - kH + 1

  return GraphNode(:conv2d, (x, W, pad_node), zeros(out_W, out_H, c_out))
end

function primal!(y::GraphNode{:flatten,1})
  x, = y.args
  y.data .= vec(x.data)
end

function adjoint!(y::GraphNode{:flatten,1})
  x, = y.args
  x.grad .+= reshape(y.grad, size(x.data)) # Bring gradient back to original shape
end

# BCE
function primal!(z::GraphNode{:bce,2})
  x, y = z.args
  ϵ = 1e-8 # Tiny epsilon to prevent exploding math
  x_safe = clamp.(x.data, ϵ, 1.0 - ϵ)

  z.data .= -(y.data .* log.(x_safe) .+ (1.0 .- y.data) .* log.(1.0 .- x_safe))
end
function adjoint!(z::GraphNode{:bce,2})
  x, y = z.args
  ϵ = 1e-8
  x_safe = clamp.(x.data, ϵ, 1.0 - ϵ)

  x.grad .-= (y.data ./ x_safe .- (1.0 .- y.data) ./ (1.0 .- x_safe)) .* z.grad
end

# Matrix Multiply
function primal!(y::GraphNode{:mul,2})
  W, x = y.args
  y.data = W.data * x.data
end
function adjoint!(y::GraphNode{:mul,2})
  W, x = y.args
  W.grad .+= y.grad * x.data'
  x.grad .+= W.data' * y.grad
end

# ReLU
function primal!(y::GraphNode{:relu,1})
  x, = y.args
  y.data .= max.(0, x.data)
end
function adjoint!(y::GraphNode{:relu,1})
  x, = y.args
  x.grad .+= (x.data .> 0) .* y.grad
end

function primal!(y::GraphNode{:softmax,1})
  x, = y.args
  shifted = x.data .- maximum(x.data)
  exp_x = exp.(shifted)
  y.data .= exp_x ./ sum(exp_x)
end
function adjoint!(y::GraphNode{:softmax,1})
  x, = y.args
  p = y.data
  dy = y.grad
  x.grad .+= p .* (dy .- sum(p .* dy))
end

# Addition
function primal!(z::GraphNode{:add,2})
  x, y = z.args
  z.data .= x.data .+ y.data
end
function adjoint!(z::GraphNode{:add,2})
  x, y = z.args
  x.grad .+= z.grad
  y.grad .+= z.grad
end

# Sigmoid
function primal!(y::GraphNode{:sigmoid,1})
  x, = y.args
  y.data .= 1 ./ (1 .+ exp.(-x.data))
end
function adjoint!(y::GraphNode{:sigmoid,1})
  x, = y.args
  x.grad .+= (y.data .* (1 .- y.data)) .* y.grad
end

# CrossEntropy
function primal!(z::GraphNode{:cce,2})
  x, y = z.args
  ϵ = 1e-8 # epsilon dla stabilności numerycznej (zapobiega log(0))
  x_safe = clamp.(x.data, ϵ, 1.0)

  # Wzór: -sum(y_true * log(y_pred))
  z.data .= -sum(y.data .* log.(x_safe))
end
function adjoint!(z::GraphNode{:cce,2})
  x, y = z.args
  ϵ = 1e-8
  x_safe = clamp.(x.data, ϵ, 1.0)

  # Pochodna: - y_true / y_pred * dz
  x.grad .-= (y.data ./ x_safe) .* z.grad[1]
end

# MaxPool
function primal!(y::GraphNode{:maxpool2x2,1})
  x, = y.args
  W, H, C = size(x.data)

  for c in 1:C
    for i in 1:div(W, 2)
      for j in 1:div(H, 2)
        # Get max value in block
        block = x.data[2i-1:2i, 2j-1:2j, c]
        y.data[i, j, c] = maximum(block)
      end
    end
  end
end
function adjoint!(y::GraphNode{:maxpool2x2,1})
  x, = y.args
  W, H, C = size(x.data)

  for c in 1:C
    for i in 1:div(W, 2)
      for j in 1:div(H, 2)
        block = x.data[2i-1:2i, 2j-1:2j, c]
        max_val = maximum(block)

        added = false
        for bi in 1:2
          for bj in 1:2
            if !added && block[bi, bj] == max_val
              x.grad[2i-2+bi, 2j-2+bj, c] += y.grad[i, j, c]
              added = true
            end
          end
        end
      end
    end
  end
end

# Dropout
function primal!(y::GraphNode{:dropout,3})
  x, mask, p_node = y.args
  p = p_node.data[1]
  global IS_TRAINING

  if IS_TRAINING
    mask.data .= rand(size(x.data)...) .> p

    y.data .= (x.data .* mask.data) ./ (1.0 - p)
  else
    y.data .= x.data
  end
end
function adjoint!(y::GraphNode{:dropout,3})
  x, mask, p_node = y.args
  p = p_node.data[1]
  global IS_TRAINING

  if IS_TRAINING
    # Gradient only flows through "active" neurons,
    x.grad .+= (y.grad .* mask.data) ./ (1.0 - p)
  else
    # Even though we are not training in adjoint!,
    # the gradient flows through normally
    x.grad .+= y.grad
  end
end

# Author: Seif_Shebl
# URL: https://discourse.julialang.org/t/what-is-julias-im2col/14066/9
@inline function im2col(A, n, m)
    M, N = size(A)
    B = Array{eltype(A)}(undef, m*n, (M-m+1)*(N-n+1))
    indx = reshape(1:M*N, M, N)[1:M-m+1, 1:N-n+1]
    for (i, value) in enumerate(indx)
        for j = 0:n-1
            @views B[(i-1)*m*n+j*m+1:(i-1)*m*n+(j+1)*m] = A[value+j*M:value+m-1+j*M]
        end
    end
    return B
end

@inline function col2im(B, M, N, n, m)
    A = zeros(eltype(B), M, N)
    indx = reshape(1:M*N, M, N)[1:M-m+1, 1:N-n+1]
    for (i, value) in enumerate(indx)
        for j = 0:n-1
            @views A[value+j*M:value+m-1+j*M] .+= B[(i-1)*m*n+j*m+1:(i-1)*m*n+(j+1)*m]
        end
    end
    return A
end

# Convolution
function primal!(y::GraphNode{:conv2d,3})
  x, W, pad_node = y.args
  img = x.data
  ker = W.data
  pad = Int(pad_node.data[1])

  img_W, img_H, C_in = size(img)
  kW, kH, _, C_out = size(ker)

  # Padding
  if pad > 0
    padded_img = zeros(img_W + 2pad, img_H + 2pad, C_in)
    padded_img[pad+1:pad+img_W, pad+1:pad+img_H, :] = img
  else
    padded_img = img
  end

  out_W, out_H, _ = size(y.data)

  # Reshape kernel for matrix multiplication: (C_out, kW * kH * C_in)
  ker_reshaped = reshape(ker, kW * kH * C_in, C_out)'

  # Apply im2col to each channel and stack into columns: (kW * kH * C_in, out_W * out_H)
  col_img = zeros(eltype(img), kW * kH * C_in, out_W * out_H)
  for c in 1:C_in
      col_img[(c-1)*kW*kH + 1 : c*kW*kH, :] = im2col(padded_img[:, :, c], kW, kH)
  end

  # A single Matrix Multiplication!
  out_col = ker_reshaped * col_img

  # Reshape the output back to (out_W, out_H, C_out)
  for c_o in 1:C_out
      y.data[:, :, c_o] .= reshape(out_col[c_o, :], out_W, out_H)
  end
end
function adjoint!(y::GraphNode{:conv2d,3})
  x, W, pad_node = y.args
  img = x.data
  ker = W.data
  pad = Int(pad_node.data[1])

  img_W, img_H, C_in = size(img)
  kW, kH, _, C_out = size(ker)
  out_W, out_H, _ = size(y.data)

  # Padding setup
  if pad > 0
    padded_img = zeros(img_W + 2pad, img_H + 2pad, C_in)
    padded_img[pad+1:pad+img_W, pad+1:pad+img_H, :] .= img
  else
    padded_img = img
  end

  # Flatten dy: (C_out, out_W * out_H)
  dy_reshaped = zeros(eltype(y.grad), C_out, out_W * out_H)
  for c_o in 1:C_out
      dy_reshaped[c_o, :] = vec(y.grad[:, :, c_o])
  end

  # Reconstruct col_img for kernel gradient: (kW * kH * C_in, out_W * out_H)
  col_img = zeros(eltype(img), kW * kH * C_in, out_W * out_H)
  for c in 1:C_in
      col_img[(c-1)*kW*kH + 1 : c*kW*kH, :] .= im2col(padded_img[:, :, c], kW, kH)
  end

  # 1. Gradient with respect to Kernel (W)
  # dW = dy * X^T
  dW_reshaped = dy_reshaped * col_img'
  W.grad .+= reshape(dW_reshaped', kW, kH, C_in, C_out)

  # 2. Gradient with respect to Input (X)
  # dX_col = W^T * dy
  ker_reshaped = reshape(ker, kW * kH * C_in, C_out)'
  dx_col = ker_reshaped' * dy_reshaped

  # Reconstruct the padded gradient image using col2im
  padded_grad = zeros(eltype(img), size(padded_img)...)
  for c in 1:C_in
      padded_grad[:, :, c] = col2im(dx_col[(c-1)*kW*kH + 1 : c*kW*kH, :], 
                                    img_W + 2pad, img_H + 2pad, kW, kH)
  end

  # Remove padding before adding to x.grad
  if pad > 0
    x.grad .+= padded_grad[pad+1:pad+img_W, pad+1:pad+img_H, :]
  else
    x.grad .+= padded_grad
  end
end
