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

function chain(operators)
  function flatten(x::Tuple)
    y = Vector{Operator}()
    for v in x
      v isa Tuple ? append!(y, v) : push!(y, v)
    end
    return y
  end
  return flatten(operators)
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

  z.data = -(y.data .* log.(x_safe) .+ (1.0 .- y.data) .* log.(1.0 .- x_safe))
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
  W.grad += y.grad * x.data'
  x.grad += W.data' * y.grad
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
  z.data = x.data .+ y.data
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
    # Even though we are not training, 
    # the gradient flows through normally
    x.grad .+= y.grad
  end
end
# Forward pass (przebieg w przód) dla naiwnego splotu
function primal!(y::GraphNode{:conv2d,3})
  x, W, pad_node = y.args
  img = x.data
  ker = W.data
  pad = Int(pad_node.data[1])

  img_W, img_H, C_in = size(img)
  kW, kH, _, C_out = size(ker)

  # Obsługa brzegu (Zero-padding)
  if pad > 0
    padded_img = zeros(img_W + 2pad, img_H + 2pad, C_in)
    padded_img[pad+1:pad+img_W, pad+1:pad+img_H, :] = img
  else
    padded_img = img
  end

  out_W, out_H, _ = size(y.data)

  fill!(y.data, 0.0)

  # Naiwna implementacja pętlami
  fill!(y.data, 0.0)
  Threads.@threads for c_o in 1:C_out
    for c_i in 1:C_in
      for i in 1:out_W
        for j in 1:out_H
          # Dodane @views
          @views block = padded_img[i:i+kW-1, j:j+kH-1, c_i]
          @views y.data[i, j, c_o] += sum(block .* ker[:, :, c_i, c_o])
        end
      end
    end
  end
end

# Backward pass (przebieg w tył - propagacja wsteczna gradientu) dla naiwnego splotu
function adjoint!(y::GraphNode{:conv2d,3})
  x, W, pad_node = y.args
  img = x.data
  ker = W.data
  pad = Int(pad_node.data[1])

  img_W, img_H, C_in = size(img)
  kW, kH, _, C_out = size(ker)

  # Przygotowanie struktur na wejście uwzględniających padding
  if pad > 0
    padded_img = zeros(img_W + 2pad, img_H + 2pad, C_in)
    padded_img[pad+1:pad+img_W, pad+1:pad+img_H, :] = img
    padded_grad = zeros(img_W + 2pad, img_H + 2pad, C_in)
  else
    padded_img = img
    padded_grad = zeros(size(img))
  end

  out_W, out_H, _ = size(y.data)

  # Przechodzenie grafu w tył przy naiwnej konwolucji
  Threads.@threads for c_o in 1:C_out
    for c_i in 1:C_in
      for i in 1:out_W
        for j in 1:out_H
          dy = y.grad[i, j, c_o]

          # Dodane @views dla bloków obrazu, gradientu i jadra
          @views block = padded_img[i:i+kW-1, j:j+kH-1, c_i]
          @views w_grad_view = W.grad[:, :, c_i, c_o]
          @views p_grad_view = padded_grad[i:i+kW-1, j:j+kH-1, c_i]
          @views ker_view = ker[:, :, c_i, c_o]

          # Mnożenie w miejscu bez tworzenia nowych tablic (Broadcast)
          w_grad_view .+= block .* dy
          p_grad_view .+= ker_view .* dy
        end
      end
    end
  end

  # Odrzucamy padding z obliczonego gradientu wejściowego
  if pad > 0
    x.grad .+= padded_grad[pad+1:pad+img_W, pad+1:pad+img_H, :]
  else
    x.grad .+= padded_grad
  end
end
