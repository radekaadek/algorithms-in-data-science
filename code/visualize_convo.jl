using Base: sleep

function visualize_conv_forward_complex()
  # 1. Setup Data: 5x5 Input
  X = [
    10 15 20 25 30;
    12 17 22 27 32;
    14 19 24 29 34;
    16 21 26 31 36;
    18 23 28 33 38
  ]

  # 3x3 Kernel (Sharpen/Edge-ish filter)
  K = [
    0 -1 0;
    -1 5 -1;
    0 -1 0
  ]

  # Dimensions
  M, N = size(X)
  m, n = size(K)
  out_H, out_W = M - m + 1, N - n + 1
  num_windows = out_H * out_W
  patch_size = m * n

  # 2. Pre-allocate
  im2col_mat = fill(0, patch_size, num_windows)
  K_row = reshape(K, 1, patch_size) # Flattened Kernel
  out_mat = fill(0, out_H, out_W)

  # ANSI Colors
  GREEN = "\033[1;32m"
  CYAN = "\033[1;36m"
  YELLOW = "\033[1;33m"
  MAGENTA = "\033[1;35m"
  RESET = "\033[0m"
  CLEAR = "\033[2J\033[H"

  step = 1
  for j in 1:out_W
    for i in 1:out_H
      print(CLEAR)
      println("$(MAGENTA)==================================================$(RESET)")
      println("     COMPLEX IM2COL: 5x5 Input ⊗ 3x3 Kernel       ")
      println("$(MAGENTA)==================================================$(RESET)\n")

      # --- DISPLAY INPUT & KERNEL ---
      println("Input Image X (5x5):            Kernel K (3x3):")
      for r in 1:M
        for c in 1:N
          if r >= i && r < i + m && c >= j && c < j + n
            print(GREEN, lpad(X[r, c], 4), RESET)
          else
            print(lpad(X[r, c], 4))
          end
        end
        if r <= m
          print("          ")
          for kc in 1:n
            print(CYAN, lpad(K[r, kc], 4), RESET)
          end
        end
        println()
      end
      println()

      # --- EXTRACT & UPDATE IM2COL ---
      block = X[i:i+m-1, j:j+n-1]
      col_vec = vec(block)
      im2col_mat[:, step] = col_vec

      println("1. im2col Matrix ($(patch_size)x$(num_windows)):")
      for r in 1:patch_size
        for c in 1:num_windows
          if c == step
            print(GREEN, lpad(im2col_mat[r, c], 3), RESET)
          elseif c < step
            print(lpad(im2col_mat[r, c], 3))
          else
            print("  .")
          end
        end
        println()
      end
      println()

      # --- RESTORED MULTIPLICATION STEP ---
      dot_product = sum(K_row[1, :] .* col_vec)
      out_mat[i, j] = dot_product

      println("2. Matrix Multiplication (K_row * im2col_col):")
      print("K_row: [", CYAN)
      print(join(lpad.(K_row[1, :], 3), ""))
      print(RESET, " ]\n")

      print("X_col: [", GREEN)
      print(join(lpad.(col_vec, 3), ""))
      print(RESET, " ]^T\n")

      # Show the calculation breakdown
      products = K_row[1, :] .* col_vec
      println("Calc:  ", join(products, " + "), " = ", YELLOW, dot_product, RESET)
      println()

      # --- DISPLAY OUTPUT ---
      println("3. Output Feature Map (3x3):")
      for r in 1:out_H
        for c in 1:out_W
          if r == i && c == j
            print(YELLOW, lpad(out_mat[r, c], 5), RESET)
          elseif (c < j) || (c == j && r < i)
            print(lpad(out_mat[r, c], 5))
          else
            print("    .")
          end
        end
        println()
      end

      println("\nStep $step of $num_windows | Window at ($i, $j)")
      step += 1
      sleep(1.2)
    end
  end
  println("\n$(GREEN)Done!$(RESET) The dot product of the flattened kernel and the image column is the exact same as a spatial convolution.")
end

visualize_conv_forward_complex()
