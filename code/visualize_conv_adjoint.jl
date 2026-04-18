using Base: reshape

@inline function im2col(A, n, m)
  M, N = size(A)
  B = Array{eltype(A)}(undef, m * n, (M - m + 1) * (N - n + 1))
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

# Helper function to print matrix math side-by-side
function print_math_expanded(A, op, B, eq, C; cA="", cB="", cC="", reset="\033[0m")
  rA, cA_len = size(A, 1), size(A, 2)
  rB, cB_len = size(B, 1), size(B, 2)
  rC, cC_len = size(C, 1), size(C, 2)
  h = max(rA, rB, rC)

  fmt(x) = lpad(round(x, digits=1), 5)

  for i in 1:h
    # Matrix A
    if i <= rA
      print(cA, "[", join([fmt(A[i, j]) for j in 1:cA_len], " "), " ]", reset)
    else
      print(" "^(cA_len * 6 + 3))
    end

    # Operator
    if i == div(h, 2) + 1 || (h == 1 && i == 1)
      print("  $op  ")
    else
      print("     ")
    end

    # Matrix B
    if i <= rB
      print(cB, "[", join([fmt(B[i, j]) for j in 1:cB_len], " "), " ]", reset)
    else
      print(" "^(cB_len * 6 + 3))
    end

    # Equals
    if i == div(h, 2) + 1 || (h == 1 && i == 1)
      print("  $eq  ")
    else
      print("     ")
    end

    # Matrix C
    if i <= rC
      print(cC, "[", join([fmt(C[i, j]) for j in 1:cC_len], " "), " ]", reset)
    else
      print(" "^(cC_len * 6 + 3))
    end
    println()
  end
end

function visualize_conv_adjoint_expanded()
  # --- 1. Setup Data: 3x3 Input, 2x2 Kernel ---
  X = [
    1.0 2.0 3.0;
    4.0 5.0 6.0;
    7.0 8.0 9.0
  ]

  W = [
    1.0 0.0;
    0.0 -1.0
  ]

  # Mock gradient coming from the next layer (dY)
  dY = [
    1.0 0.5;
    -0.5 2.0
  ]

  M, N = size(X)
  m, n = size(W)
  out_H, out_W = M - m + 1, N - n + 1
  C_out = 1

  # --- 2. Adjoint Calculations ---
  # Flatten dY: (C_out, out_W * out_H) -> 1x4 matrix
  dy_reshaped = reshape(dY, C_out, out_W * out_H)

  # Reconstruct col_img for kernel gradient: (kW * kH * C_in, out_W * out_H) -> 4x4 matrix
  col_img = im2col(X, m, n)

  # Gradient w.r.t Kernel (dW) -> dW_reshaped = dy * X^T
  dW_reshaped = dy_reshaped * col_img'
  dW = reshape(dW_reshaped', m, n)

  # Gradient w.r.t Input (X) -> dX_col = W^T * dy
  ker_reshaped = reshape(W, m * n, C_out)'
  dx_col = ker_reshaped' * dy_reshaped

  # Reconstruct the gradient image using col2im
  dX = col2im(dx_col, M, N, m, n)

  # --- 3. Visualization ---
  GREEN = "\033[1;32m"
  CYAN = "\033[1;36m"
  YELLOW = "\033[1;33m"
  MAGENTA = "\033[1;35m"
  RESET = "\033[0m"

  println("$(MAGENTA)======================================================================$(RESET)")
  println("          EXPANDED ADJOINT VISUALIZATION: Conv2D Backward Pass          ")
  println("$(MAGENTA)======================================================================$(RESET)\n")

  println("$(YELLOW)--- 1. FORWARD PASS STATE ---$(RESET)")
  println("Input Image X (3x3):       Kernel W (2x2):       Upstream Grad dY (2x2):")
  for r in 1:M
    for c in 1:N
      print(GREEN, lpad(X[r, c], 5), RESET)
    end
    print("      ")
    if r <= m
      for kc in 1:n
        print(CYAN, lpad(W[r, kc], 5), RESET)
      end
      print("            ")
    else
      print("                        ")
    end
    if r <= out_H
      for yc in 1:out_W
        print(YELLOW, lpad(dY[r, yc], 5), RESET)
      end
    end
    println()
  end
  println()

  println("$(YELLOW)--- 2. GRADIENT W.R.T KERNEL (dW) ---$(RESET)")
  println("Formula: $(CYAN)dW_reshaped$(RESET) = $(YELLOW)dy_reshaped$(RESET) * $(GREEN)col_img^T$(RESET)\n")

  print_math_expanded(dy_reshaped, "*", col_img', "=", dW_reshaped,
    cA=YELLOW, cB=GREEN, cC=CYAN)

  println("\nReshaping $(CYAN)dW_reshaped$(RESET) back to 2x2 yields $(CYAN)dW$(RESET):")
  for r in 1:m
    for c in 1:n
      print(CYAN, lpad(dW[r, c], 7), RESET)
    end
    println()
  end
  println()

  println("$(YELLOW)--- 3. GRADIENT W.R.T INPUT (dX) ---$(RESET)")
  println("Formula: $(GREEN)dx_col$(RESET) = $(CYAN)W_reshaped^T$(RESET) * $(YELLOW)dy_reshaped$(RESET)\n")

  print_math_expanded(ker_reshaped', "*", dy_reshaped, "=", dx_col,
    cA=CYAN, cB=YELLOW, cC=GREEN)

  println("\nApplying $(MAGENTA)col2im$(RESET) on $(GREEN)dx_col$(RESET) overlaps the gradients to yield $(GREEN)dX (3x3)$(RESET):")
  for r in 1:M
    for c in 1:N
      print(GREEN, lpad(dX[r, c], 7), RESET)
    end
    println()
  end
  println("\n$(MAGENTA)Done!$(RESET) Gradients successfully expanded into matrix operations.")
end

visualize_conv_adjoint_expanded()
