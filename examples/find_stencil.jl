stencil = [-9, 125, -2250, 2250, -125, 9]
stencil = vcat(stencil, fill(0, length(stencil)))

rows = map(i -> circshift(stencil, i-1), eachindex(stencil))

mat = stack(rows)'
a = (mat * mat)[1, :]


a // 1920^2

1920^2
