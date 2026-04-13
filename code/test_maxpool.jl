include("autodiff.jl")
include("layers.jl")

# 1. Tworzymy testowy wektor 4x4x1
test_input = reshape(Float64[
    1 2 5 6;
    3 4 7 8;
    9 10 13 14;
    11 12 15 16
  ], 4, 4, 1)

# Tworzymy węzeł (tensor) do grafu
x = GraphNode(test_input, true)

# 2. Definiujemy warstwę i przypinamy ją do grafu
mp = maxpool()
y = mp(x)

# 3. Test przejścia w przód (Forward pass)
primal!(y)
println("=== WEJŚCIE (4x4) ===")
display(x.data[:, :, 1])
println("\n=== WYJŚCIE MAXPOOL (2x2) ===")
display(y.data[:, :, 1])
# Powinno wypisać maksimum z każdego bloku 2x2:
# [4.0, 8.0; 12.0, 16.0]

# 4. Test propagacji wstecznej (Backward pass)
y.grad = fill(1.0, 2, 2, 1) # symulujemy gradient z kolejnej warstwy
adjoint!(y)
println("\n=== GRADIENT NA WEJŚCIU (Powinien być 1.0 tam gdzie było maksimum) ===")
display(x.grad[:, :, 1])
