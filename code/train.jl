# main.jl
include("layers.jl")
using MLDatasets, Random

train_data = MLDatasets.FashionMNIST(split=:train)
test_data  = MLDatasets.FashionMNIST(split=:test)

# Network Setup
net = chain((
    dense(784 => 16, relu),
    dense(16 => 10, sigmoid),
))

input  = tensor(784)()
target = tensor(10)()
output = net(input)
loss   = bce(output, target)
model  = graph(loss)

# Helpers
function prep_data(data)
    N = length(data.targets)
    # Flatten the 28x28 images into 784xN vectors
    xs = reshape(data.features, 784, N)
    
    # One-hot encode the 10 classes
    ys = zeros(10, N)
    for i in 1:N
        ys[data.targets[i] + 1, i] = 1.0 
    end
    
    # Cast to Float64 to ensure compatibility with your custom layers
    return Float64.(xs), Float64.(ys)
end

function test(model, inputs, targets, input_node, output_node)
    correct = 0
    N = size(inputs, 2)
    for i in 1:N
        forward!(model, input_node => inputs[:, i])
        
        # Determine the predicted class and the true class
        pred_class = argmax(output_node.data)
        true_class = argmax(targets[:, i])
        
        if pred_class == true_class
            correct += 1
        end
    end
    println("Accuracy: ", round((correct / N) * 100, digits=2), "%")
end

function train!(model, batch_indices, inputs, targets, input_node, target_node; learning_rate=1e-2)
    L = 0.0
    for i in batch_indices
        zerograd!(model)
        forward!(model, input_node => inputs[:, i], target_node => targets[:, i])
        backward!(model)
        L += sum(model[end].data)
        optimize!(model, learning_rate)
    end
    return L
end

# Execution
inputs, targets = prep_data(train_data)
test_inputs, test_targets = prep_data(test_data)

# Taking a 500-sample batch for training (adjust this if you want to train on the full 60,000)
batch = collect(1:500)

println("[x] Random model on test data:")
test(model, test_inputs, test_targets, input, output)

println("\n[x] Training...")
for i in 1:10
    L = train!(model, batch, inputs, targets, input, target, learning_rate=1e-2)
    println("[+] Epoch $i - Loss: ", round(L, digits=4))
end

println("\n[x] Final model on test data:")
test(model, test_inputs, test_targets, input, output)
