# layers.jl
include("autodiff.jl")

abstract type Operator end
const Chain = Vector{Operator}

struct Sigmoid <: Operator end
struct ReLU <: Operator end
struct Dense <: Operator
  insize::Int64
  outsize::Int64
end

relu() = ReLU()
sigmoid() = Sigmoid()
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
  W = GraphNode(randn(y.outsize, y.insize), true)
  b = GraphNode(randn(y.outsize), true)
  mul = GraphNode(:mul, (W, x), zeros(y.outsize))
  return GraphNode(:add, (mul, b), zeros(y.outsize))
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
