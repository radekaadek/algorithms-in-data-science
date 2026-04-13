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

relu() = ReLU()
sigmoid() = Sigmoid()
maxpool() = MaxPool()
flatten() = Flatten()
dropout(p::Float64) = Dropout(p)
dense(pair::Pair{Int64,Int64}) = Dense(first(pair), last(pair))
dense(pair::Pair{Int64,Int64}, activation) = tuple(dense(pair), activation())

abstract type Loss end
struct BinaryCrossEntropy <: Loss end
bce(output, target) = BinaryCrossEntropy()(output, target)

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

        # Add gradient where the max value is
        for bi in 1:2
          for bj in 1:2
            if block[bi, bj] == max_val
              x.grad[2i-2+bi, 2j-2+bj, c] += y.grad[i, j, c]
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
