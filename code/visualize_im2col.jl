# Run this directly in your Julia REPL
using Base: sleep

function visualize_im2col()
  # Define input matrix (4x4) and kernel size (2x2)
  M, N = 4, 4
  m, n = 2, 2
  A = reshape(1:(M*N), M, N)

  # Calculate output dimensions
  out_cols = (M - m + 1) * (N - n + 1)
  out_rows = m * n
  B = zeros(Int, out_rows, out_cols)

  # ANSI color codes for visualization
  GREEN = "\033[1;32m"
  RESET = "\033[0m"
  CLEAR_SCREEN = "\033[2J"
  CURSOR_TOP_LEFT = "\033[H"

  print(CLEAR_SCREEN)

  col_idx = 1
  # Loop over the columns of the input matrix (Column-major sliding window)
  for j in 1:(N-n+1)
    for i in 1:(M-m+1)
      print(CURSOR_TOP_LEFT)

      println("=========================================")
      println("        im2col Visualization           ")
      println("=========================================\n")
      println("Input Matrix A ($M x $N) | Kernel ($m x $n)\n")

      # 1. Print Input Matrix A
      for r in 1:M
        for c in 1:N
          # Check if current cell is within the active sliding window
          if r >= i && r < i + m && c >= j && c < j + n
            print(GREEN, lpad(A[r, c], 4), RESET)
          else
            print(lpad(A[r, c], 4))
          end
        end
        println()
      end

      println("\n-----------------------------------------")
      println("\nOutput Matrix B ($out_rows x $out_cols)\n")

      # Extract the block and flatten it to a column
      # Note: Julia is column-major, so vec() flattens down columns first
      block = A[i:i+m-1, j:j+n-1]
      B[:, col_idx] = vec(block)

      # 2. Print Output Matrix B
      for r in 1:out_rows
        for c in 1:out_cols
          if c == col_idx
            # Highlight the column currently being filled
            print(GREEN, lpad(B[r, c], 4), RESET)
          elseif c < col_idx
            # Print previously filled columns normally
            print(lpad(B[r, c], 4))
          else
            # Print placeholders for future columns
            print("   -")
          end
        end
        println()
      end

      println("\n\nStep $col_idx of $out_cols")
      println("Mapping block A[$i:$(i+m-1), $j:$(j+n-1)] -> B[:, $col_idx]")

      col_idx += 1

      # Pause to create an animation effect
      sleep(0.8)
    end
  end

  println("\nVisualization Complete!")
end

# Execute the visualization
visualize_im2col()
