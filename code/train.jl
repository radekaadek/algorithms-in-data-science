include("layers.jl")
using MLDatasets, Random, ProgressMeter, CairoMakie, Profile, ProfileView, StatProfilerHTML

train_data = MLDatasets.FashionMNIST(split=:train)
test_data = MLDatasets.FashionMNIST(split=:test)

# Helpers
function prep_data(data)
  N = length(data.targets)
  xs = reshape(data.features, 784, N)
  ys = zeros(10, N)
  for i in 1:N
    ys[data.targets[i]+1, i] = 1.0
  end
  return Float64.(xs), Float64.(ys)
end

function test(model, inputs, targets, input_node, output_node; desc="Testing: ")
  global IS_TRAINING = false
  correct = 0
  N = size(inputs, 2)

  prog = Progress(N; dt=0.01, desc=desc, barlen=40, color=:green)

  for i in 1:N
    forward!(model, input_node => reshape(inputs[:, i], 28, 28, 1))
    pred_class = argmax(output_node.data)
    true_class = argmax(targets[:, i])

    if pred_class == true_class
      correct += 1
    end
    next!(prog)
  end
  acc = (correct / N) * 100
  global IS_TRAINING = true
  return acc
end

function train!(model, all_indices, inputs, targets, input_node, target_node; learning_rate=1e-2, epoch=1, total_epochs=3, batchsize=10)
  L = 0.0
  shuffle!(all_indices)

  # Partition indices into mini-batches
  batches = collect(Iterators.partition(all_indices, batchsize))
  prog = Progress(length(batches); dt=0.01, desc="Epoch $epoch/$total_epochs: ", barlen=40, color=:cyan)

  # Extract only the weight nodes so we can manually accumulate gradients
  weight_nodes = [n for n in model if n isa GraphWeight]

  for batch in batches
    # Initialize zero accumulators for this batch
    weight_grads = [zeros(size(w.data)) for w in weight_nodes]
    batch_loss = 0.0

    for i in batch
      zerograd!(model) # Zero gradients for intermediate nodes
      forward!(model, input_node => reshape(inputs[:, i], 28, 28, 1), target_node => targets[:, i])
      backward!(model)
      batch_loss += sum(model[end].data)

      # Accumulate weight gradients from this sample
      for (w, wg) in zip(weight_nodes, weight_grads)
        wg .+= w.grad
      end
    end

    # Overwrite the graph's weight gradients with the batch average
    for (w, wg) in zip(weight_nodes, weight_grads)
      w.grad .= wg ./ length(batch)
    end

    # Optimize using the averaged batch gradients
    optimize!(model, learning_rate)

    L += batch_loss
    next!(prog)
  end

  return L / length(all_indices)
end

net = chain((
  Conv((3, 3), 1 => 6, pad=1, bias=false),
  maxpool(),
  Conv((3, 3), 6 => 16, pad=1, bias=false),
  maxpool(),
  flatten(),
  dense(784 => 84, relu),
  dropout(0.4),
  dense(84 => 10, softmax),
))

input = tensor(28, 28, 1)()
target = tensor(10)()
output = net(input)
loss = cce(output, target)
model = graph(loss)

# Execution
inputs, targets = prep_data(train_data)
test_inputs, test_targets = prep_data(test_data)

all_train_indices = collect(1:size(inputs, 2))
all_test_indices = collect(1:size(test_inputs, 2))

settings = (
  eta=1e-2,
  epochs=3,
  batchsize=10,
)

println("[x] Random model on full test data:")
init_acc = test(model, test_inputs, test_targets, input, output, desc="Testing Init: ")
println("Accuracy: ", round(init_acc, digits=2), "%\n")

println("[x] Training...")
accuracy = zeros(settings.epochs, 2)

for epoch in 1:settings.epochs
  L = train!(model, all_train_indices, inputs, targets, input, target,
    learning_rate=settings.eta, epoch=epoch, total_epochs=settings.epochs, batchsize=settings.batchsize)

  # Calculate train/test accuracy to log and plot
  train_acc = test(model, inputs, targets, input, output, desc="Eval Train:   ")
  test_acc = test(model, test_inputs, test_targets, input, output, desc="Eval Test:    ")

  println("↳ End of Epoch $epoch - Avg Loss: ", round(L, digits=4),
    " | Train Acc: ", round(train_acc, digits=2), "%",
    " | Test Acc: ", round(test_acc, digits=2), "%\n")

  accuracy[epoch, 1] = train_acc
  accuracy[epoch, 2] = test_acc
end
