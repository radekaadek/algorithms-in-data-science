include("autodiff.jl")
include("layers.jl")
using Random

# Ustalamy ziarno dla powtarzalności testu
Random.seed!(42)

# 1. Przygotowanie danych testowych (wektor 10-elementowy samych jedynek)
raw_data = ones(10)
x = GraphNode(raw_data, false)

# 2. Definicja warstwy Dropout (p=0.5)
# Przy p=0.5 spodziewamy się, że ok. połowa wartości zostanie wyzerowana,
# a pozostałe zostaną pomnożone przez 1/(1-0.5) = 2.0
drp = dropout(0.5)
y = drp(x)

println("=== TEST DROPOUT (p=0.5) ===")

# 3. Test w trybie TRENINGU
global IS_TRAINING = true
primal!(y)
println("\nTryb: TRENING (IS_TRAINING = true)")
println("Wejście: ", x.data)
println("Wyjście: ", y.data)
println("Suma wyjść: ", sum(y.data), " (Oczekiwana suma bliska 10.0 dzięki skalowaniu)")

# 4. Test w trybie EWALUACJI
global IS_TRAINING = false
primal!(y)
println("\nTryb: TEST (IS_TRAINING = false)")
println("Wyjście: ", y.data)
println("Suma wyjść: ", sum(y.data), " (Powinna być dokładnie 10.0)")

# 5. Test propagacji wstecznej (Adjoint)
global IS_TRAINING = true
zerograd!([x, y]) # Resetujemy gradienty
y.grad .= 1.0     # Symulujemy gradient przychodzący z góry (same jedynki)
adjoint!(y)

println("\n=== TEST GRADIENTU (Tryb treningowy) ===")
println("Gradient wejściowy (y.grad): ", y.grad)
println("Gradient obliczony (x.grad): ", x.grad)
println("Wyjaśnienie: Gradient powinien wynosić 2.0 tam, gdzie neuron był aktywny, i 0.0 tam, gdzie został wyzerowany.")
