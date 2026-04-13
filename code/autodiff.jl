using LinearAlgebra

mutable struct GraphNode{OP,N}
  args::NTuple{N,GraphNode}
  grad
  data
end

const GraphWeight = GraphNode{:weight,0}
const GraphTensor = GraphNode{:tensor,0}

function GraphNode(data::T, trainable=false) where T
  if trainable
    return GraphNode{:weight,0}((), zero(data), data)
  else
    return GraphNode{:tensor,0}((), zero(data), data)
  end
end

function GraphNode(op::Symbol, args::Tuple, data::T) where T
  N = length(args)
  grad = similar(data)
  return GraphNode{op,N}(args, grad, data)
end

function graph(node)
  function visit!(node::GraphNode, visited, ordered)
    if !(node in visited)
      push!(visited, node)
      for arg in node.args
        visit!(arg, visited, ordered)
      end
      push!(ordered, node)
    end
    return nothing
  end
  ordered = Vector{GraphNode}()
  visited = Set{GraphNode}()
  visit!(node, visited, ordered)
  return ordered
end

function zerograd!(order::Vector{GraphNode})
  for node in order
    node.grad .= 0
  end
end

# Default passes for leaf nodes
primal!(::GraphTensor) = nothing
primal!(::GraphWeight) = nothing
adjoint!(::GraphTensor) = nothing
adjoint!(::GraphWeight) = nothing

function forward!(order::Vector{GraphNode}, pairs...)
  for (tensor, data) in pairs
    tensor.data .= data
  end
  for node in order
    primal!(node)
  end
end

function backward!(order::Vector{GraphNode})
  seed = last(order)
  seed.grad .= 1
  for node in reverse(order)
    adjoint!(node)
  end
end

function optimize!(graph, η)
  for node in graph
    if node isa GraphWeight
      node.data .-= η * node.grad
    end
  end
end
