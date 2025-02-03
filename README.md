# RightLookLU.jl

Threaded LU decomposition in Julia.

This package decomposes a matrix into it's `L` and `U` factors. It implements the API of UMFPACK factorisations, i.e. `lu!(old_lu, new_matrix)` where in UMFPACK's case, `new_matrix` 
has the same sparsity pattern as the one used to create `old_lu`.
