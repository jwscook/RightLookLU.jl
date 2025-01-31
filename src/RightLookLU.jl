module RightLookLU

using Base.Threads, LinearAlgebra, SparseArrays
using ChunkSplitters
using BlockArrays

lsolve!(A, L::AbstractSparseArray) = (A .= L \ A) # can't mutate L
lsolve!(A, L) = ldiv!(A, lu(L), A) # could use a work array here in lu!
function lsolve!(A, L, work)
  W = view(work, 1:size(L, 1), 1:size(L, 2))
  copyto!(W, L) # W .= L
  #LoopVectorization.vmapntt!(identity, W, L)
  luW = lu!(W, Val(true); check=false) # lu!(W, NoPivot(); check=false) calls generic lu!
  ldiv!(luW, A)
end
rsolve!(A, U::AbstractSparseArray) = (A .= A / U) # can't mutate U
rsolve!(A, U) = rdiv!(A, lu(U)) # could use a work array here in lu!
function rsolve!(A, U, work)
  W = view(work, 1:size(U, 1), 1:size(U, 2))
  copyto!(W, U) # W .= U
  #LoopVectorization.vmapntt!(identity, W, U)
  luW = lu!(W, Val(true); check=false) # lu!(W, NoPivot(); check=false) calls generic lu!
  rdiv!(A, luW)
end

struct RightLookLU{T, M<:AbstractMatrix{T}}
  A::M
  ntiles::Int
  rowindices::Vector{UnitRange{Int64}}
  colindices::Vector{UnitRange{Int64}}
  isempties::Matrix{Bool}
  works::Vector{Matrix{T}}
end
function RightLookLU(A::AbstractMatrix, ntiles::Int)
  rowindices = collect(chunks(1:size(A, 1); n=ntiles))
  colindices = collect(chunks(1:size(A, 2); n=ntiles))
  isempties = zeros(Bool, length(rowindices), length(colindices))
  A = BlockArray(A, [length(is) for is in rowindices], [length(js) for js in colindices])
  works = [similar(A, maximum(length(is) for is in rowindices),
                      maximum(length(js) for js in colindices)) for _ in 1:nthreads()]
  return RightLookLU(A, ntiles, rowindices, colindices, isempties, works)
end

tile(A::RightLookLU{T, M}, i, j) where {T, M<:BlockArray{T}} = blocks(A.A)[i, j]

function tile(A::RightLookLU{T, M}, i, j) where {T, M}
  is, js = A.rowindices[i], A.colindices[j]
  return view(A.A, is, js)
end

function LinearAlgebra.lu!(RL::RightLookLU, A::AbstractMatrix)
  tasks = Task[]
  for (i, is) in enumerate(RL.rowindices), (j, js) in enumerate(RL.colindices)
    RL.isempties[i, j] && continue # must have same sparsity pattern TODO check
    push!(tasks, @spawn copyto!(tile(RL, i, j), view(A, is, js)))
  end
  wait.(tasks)
  return lu!(RL)
end
function LinearAlgebra.lu!(RL::RightLookLU)
  for level in 1:RL.ntiles
    factorise!(RL, level)
  end
  return RL
end

function factorise!(A::RightLookLU, level)
  All = subtractleft!(A, level, level)
  Lll, Ull = lu!(All, NoPivot(); check=false)
  lks = Vector{Tuple{Int,Int}}()
  for k in level + 1:A.ntiles
      push!(lks, (level, k))
      push!(lks, (k, level))
  end
  @threads for it in lks
    lookright!(A, it..., Lll, Ull)
  end
  return A
end

function lookright!(A::RightLookLU, i, j, L, U)
  @assert i != j 
  Aij = subtractleft!(A, i, j)
  A.isempties[i, j] && return
  i > j && rsolve!(Aij, U, A.works[threadid()])
  i < j && lsolve!(Aij, L, A.works[threadid()])
  A.isempties[i, j] = iszero(Aij)
end

function _mul!(A, L, U)
  A .-= L * U
end
function _mul!(A::Matrix{T}, L, U) where T
  BLAS.gemm!('N', 'N', -one(T), L, U, one(T), A) # gemm!(tA, tB, alpha, A, B, beta, C) # Update C as alpha*A*B + beta*C or
end
function _mul!(A::SparseMatrixCSC, L, U)
  mul!(A, L, U, -1, true) #mul!(C, A, B, α, β); C == A * B * α + C_original * β
end

function subtractleft!(A::RightLookLU, i, j)
  Aij = tile(A, i, j)
  for k in 1:min(i, j) - 1
    (A.isempties[i, k] || A.isempties[k, j]) && continue
    Lik = tile(A, i, k)
    Ukj = tile(A, k, j)
    _mul!(Aij, Lik, Ukj)
  end
  A.isempties[i, j] = iszero(Aij)
  return Aij
end

end # module

