# Find coefficients for linear combination of δ1, δ3, δ5, etc
# such that the leading order error terms cancel out
# Note: order = 2n
function find_stencil(n)
    A = broadcast((i, j) -> (2j - 1)^2i // 1, 1:(n-1), (1:n)') # Note: Matrix of rationals
    A = vcat(A, fill(1, 1, n)) # Last row: ones (sum of coefficients)
    y = map(i -> i == n, 1:n) # Right-hand side: zeros (terms cancel out), except for last element (sum of coefficients = 1)
    x = A \ y # Solve for coefficients x
end

# Make circular matrix for d/dx, then square it to
# get laplacian stencil d^2/dx^2
function find_laplacian(weights)
    weights = map(i -> weights[i] / (2i - 1), eachindex(weights)) # Account for coarser grid sizes of δ3, δ5, etc.
    stencil = vcat(-reverse(weights), weights) # Make symmetric ("right minus left")
    stencil = vcat(stencil, fill(0, length(stencil) - 1)) # Pad with zeros so we get all combinations
    rows = map(i -> circshift(stencil, i - 1), eachindex(stencil)) # Make circular matrix
    mat = stack(rows)' # Assemble matrix for d/dx
    a = (mat * mat) # Square to get laplacian stencil
    a[1, :] # Take stencil from first row
end

find_stencil(1) # Second order
find_stencil(2) # Fourth order
find_stencil(3) # Sixth order
find_stencil(4) # Eighth order
find_stencil(5) # Tenth order

find_laplacian(find_stencil(1))
find_laplacian(find_stencil(2))
find_laplacian(find_stencil(3))
find_laplacian(find_stencil(4))
find_laplacian(find_stencil(5))

find_laplacian(find_stencil(5)) * 1.0 .|> x -> round(x; digits = 4)
