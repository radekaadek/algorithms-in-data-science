using Base: sleep

function visualize_col2im_step_by_step()
    # The flattened gradient matrix (dx_col) from the backward pass
    dx_col = [
         1.0  -0.5   0.5   2.0;
         0.0   0.0   0.0   0.0;
         0.0   0.0   0.0   0.0;
        -1.0   0.5  -0.5  -2.0
    ]

    # Dimensions
    M, N = 3, 3 # Image size
    m, n = 2, 2 # Kernel size
    out_H, out_W = M - m + 1, N - n + 1 # Number of sliding windows

    # Target matrix for accumulation
    dX = zeros(M, N)

    # ANSI Colors
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    CYAN = "\033[1;36m"
    MAGENTA = "\033[1;35m"
    RESET = "\033[0m"
    CLEAR = "\033[2J\033[H"

    step = 1
    
    # Loop over the spatial dimensions exactly like the forward pass
    for j in 1:out_W
        for i in 1:out_H
            print(CLEAR)
            println("$(MAGENTA)==================================================$(RESET)")
            println("          col2im OVERLAP ACCUMULATION           ")
            println("$(MAGENTA)==================================================$(RESET)\n")

            println("$(CYAN)Full dx_col matrix (4x4):$(RESET)")
            for r in 1:size(dx_col, 1)
                for c in 1:size(dx_col, 2)
                    if c == step
                        print(CYAN, lpad(dx_col[r, c], 6), RESET)
                    else
                        print(lpad(dx_col[r, c], 6))
                    end
                end
                println()
            end
            println("\n--------------------------------------------------")

            println("$(YELLOW)Step $step: Processing Column $step (Window at $i, $j)$(RESET)")
            col = dx_col[:, step]
            
            patch = reshape(col, m, n)
            println("\n1. Reshaped column into $(m)x$(n) patch:")
            for r in 1:m
                for c in 1:n
                    print(YELLOW, lpad(patch[r, c], 6), RESET)
                end
                println()
            end

            println("\n2. Adding patch to dX at window position ($i, $j)...")

            # The actual accumulation math
            dX[i:i+m-1, j:j+n-1] .+= patch

            println("\n$(GREEN)Current State of dX (3x3):$(RESET)")
            for r in 1:M
                for c in 1:N
                    # Highlight the region that was just updated
                    if r >= i && r < i+m && c >= j && c < j+n
                        print(GREEN, lpad(dX[r, c], 8), RESET)
                    else
                        print(lpad(dX[r, c], 8))
                    end
                end
                println()
            end

            step += 1
            sleep(2.5) # Pause to let you read the step before continuing
        end
    end

    println("\n$(MAGENTA)Done!$(RESET) Notice how the center pixel accumulated overlapping gradients.")
end

visualize_col2im_step_by_step()
