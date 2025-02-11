module RightLookLU

using Base.Threads, LinearAlgebra, SparseArrays
using ChunkSplitters
using BlockArrays
#using HybridBlockArrays
#import HybridBlockArrays: tile

export RLLU

function luwork!(A, work)
  W = view(work, 1:size(A, 1), 1:size(A, 2))
  copyto!(W, A)
  return lu!(W, Val(true); check=false) # lu!(W, NoPivot(); check=false) calls generic lu!
end
lsolve!(A, L::AbstractSparseArray, _) = (A .= L \ A) # can't mutate L
function lsolve!(A, L, work)
  luW = luwork!(L, work)
  ldiv!(luW, A)
end
rsolve!(A, U::AbstractSparseArray, _) = (A .= A / U) # can't mutate U
function rsolve!(A, U, work)
  luW = luwork!(U, work)
  rdiv!(A, luW)
end

struct RLLU{T, M<:AbstractMatrix{T}}
  A::M
  ntiles::Int
  rowindices::Vector{UnitRange{Int64}}
  colindices::Vector{UnitRange{Int64}}
  isempties::Matrix{Bool}
  works::Vector{Matrix{T}}
  istransposed::Ref{Bool}
end
function RLLU(A::AbstractMatrix, ntiles::Int=32)
  @assert ntiles <= minimum(size(A))
  rowindices = collect(chunks(1:size(A, 1); n=ntiles))
  colindices = collect(chunks(1:size(A, 2); n=ntiles))
  isempties = zeros(Bool, length(rowindices), length(colindices))
  A = BlockArray(A, [length(is) for is in rowindices], [length(js) for js in colindices])
  #A = HybridBlockArray(A, rowindices, colindices; sparsitythreshold=0.3)
  works = [similar(A, maximum(length(is) for is in rowindices),
                      maximum(length(js) for js in colindices)) for _ in 1:nthreads()]
  return RLLU(A, ntiles, rowindices, colindices, isempties, works, Ref(false))
end
Base.size(A::RLLU) = size(A.A)
Base.size(A::RLLU, i) = size(A.A, i)

tile(A::RLLU{T}, i, j) where {T} = tile(A.A, i, j)
tile(A::BlockArray{T}, i, j) where {T} = blocks(A)[i, j]

function LinearAlgebra.lu!(RL::RLLU, A::AbstractMatrix)
  tasks = Task[]
  for (i, is) in enumerate(RL.rowindices), (j, js) in enumerate(RL.colindices)
    RL.isempties[i, j] && continue # must have same sparsity pattern TODO check
    push!(tasks, @spawn copyto!(tile(RL, i, j), view(A, is, js)))
  end
  wait.(tasks)
  return lu!(RL)
end
function LinearAlgebra.lu!(RL::RLLU)
  for level in 1:RL.ntiles
    factorise!(RL, level)
  end
  return RL
end
function LinearAlgebra.transpose!(A::RLLU)
  tmp = deepcopy(A.rowindices)
  A.rowindices .= A.colindices
  A.colindices .= tmp
  transpose!(A.A)
  A.isempties .= A.isempties'
  A.istransposed[] = !A.istransposed[]
  return A
end
function LinearAlgebra.transpose!(A::RLLU{T, <:BlockArray}) where T
  tmp = deepcopy(A.rowindices)
  A.rowindices .= deepcopy(A.colindices)
  A.colindices .= tmp
  A.A .= Matrix(transpose(A.A)) # no inplace for blockarray
  A.isempties .= A.isempties'
  A.istransposed[] = !A.istransposed[]
  return A
end

function diagonallu!(A::Matrix, _)
  L, U = lu!(A, Val(true); check=false)
  return (L, U)
end
function diagonallu!(A::SparseMatrixCSC, work)
  L, U = lu!(A, NoPivot(); check=false)
  return (L, U)
end
function factorise!(A::RLLU, level)
  All = subtractleft!(A, level, level)
  Lll, Ull = diagonallu!(All, A.works[threadid()])
  Lll = LowerTriangular(Lll) # non-allocating
  Ull = UpperTriangular(Ull) # non-allocating
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

function lookright!(A::RLLU, i, j, L, U)
  @assert i != j 
  Aij = subtractleft!(A, i, j)
  A.isempties[i, j] && return
  i > j && rsolve!(Aij, U, A.works[threadid()])
  i < j && lsolve!(Aij, L, A.works[threadid()])
  A.isempties[i, j] = iszero(Aij)
end

function _subtractleftmul!(A, L, U)
  A .-= L * U
end
function _subtractleftmul!(A::Matrix{T}, L::Matrix{T}, U::Matrix{T}) where T
  BLAS.gemm!('N', 'N', -one(T), L, U, one(T), A) # gemm!(tA, tB, alpha, A, B, beta, C) # Update C as alpha*A*B + beta*C or
end
function _subtractleftmul!(A::SparseMatrixCSC, L, U)
  mul!(A, L, U, -1, true) #mul!(C, A, B, α, β); C == A * B * α + C_original * β
end

function subtractleft!(A::RLLU, i, j)
  Aij = tile(A, i, j)
  for k in 1:min(i, j) - 1
    (A.isempties[i, k] || A.isempties[k, j]) && continue
    Lik = tile(A, i, k)
    Ukj = tile(A, k, j)
    _subtractleftmul!(Aij, Lik, Ukj)
  end
  A.isempties[i, j] = isempty(Aij) || iszero(Aij)
  return Aij
end

LinearAlgebra.:\(A::RLLU, b) = ldiv!(A, deepcopy(b))
function LinearAlgebra.ldiv!(x, A::RLLU, b)
  x .= b
  return ldiv!(A, x)
end
fastin(i, inds::UnitRange) = inds.start <= i <= inds.stop
function findtile(i::Int, indices)::Int
  for (ii, inds) in enumerate(indices)
    fastin(i, inds) && return ii
  end
  throw(BoundsError("$i not in $indices"))
  return 0
end
function tileloop!(s::Number, t::SparseMatrixCSC, b, j, trows::T, brows::T
    ) where {T<:UnitRange}
  tA = trows.start
  Δi = brows.start - tA
  cA, cZ = t.colptr[j], (t.colptr[j+1] - 1)
  tZ = trows.stop
  vt = view(t.rowval, cA:cZ)
  cA1 = searchsortedfirst(vt, tA) + cA - 1
  cZ = searchsortedlast(vt, tZ) + cA - 1
  cA = cA1
  for c in cA:cZ
    i = t.rowval[c]
    tc = t.nzval[c]
    bi = b[i + Δi]
    s = muladd(tc, bi, s)
  end
  return s
end
function tileloop!(s, t::SparseMatrixCSC, b, j, trows::T, brows::T
    ) where {T<:UnitRange}
  tA = trows.start
  Δi = brows.start - tA
  cA, cZ = t.colptr[j], (t.colptr[j+1] - 1)
  tZ = trows.stop
  vt = view(t.rowval, cA:cZ)
  cA1 = searchsortedfirst(vt, tA) + cA - 1
  cZ = searchsortedlast(vt, tZ) + cA - 1
  cA = cA1
  for c in cA:cZ
    i = t.rowval[c]
    ii = i + Δi
    for k in 1:size(b, 2)
      s[k] += t.nzval[c] * b[ii, k]
    end
  end
  return s
end
function tileloop!(s::Number, t, b, j, trows, brows)
  tv = view(t, trows, j)
  bv = view(b, brows)
  return s + BLAS.dotu(tv, bv)
end
function tileloop!(s, t, b, j, trows, brows)
  tv = view(t, trows, j)
  bv = view(b, brows, :)
  # gemm!(tA, tB, a, A, B, b, C) # C = a*A*B + b*C
  BLAS.gemm!('T', 'N', one(eltype(s)), bv, tv, one(eltype(s)), s)
  return s
end
function finalloop!(b, s, invAjj, j)
  @inbounds @simd for k in 1:size(b, 2)
    b[j, k] = (b[j, k] - s[k]) * invAjj # L[j, j] should be 1 but it shares a diagonal with U
  end
  return b
end
function finalloop!(b::AbstractVector, s::Number, invAjj, j)
  @inbounds b[j] = (b[j] - s) * invAjj
  return b
end
prepsummand(s::Number) = zero(typeof(s))
prepsummand(s) = fill!(s, 0)

function hotloopldiv!(s, A::RLLU{T}, b, rows, j, itiles, jtile, xinvAjj::Bool) where {T}
  s = prepsummand(s)
  #for i in rows, k in 1:size(b, 2); s[k] += A.A[i, j] * b[i, k]; end; return ### the default
  Δj = A.colindices[jtile].start - 1
  itile = findtile(j, A.rowindices) # not necessary if A.rowindices == A.colindices
  Δi = A.rowindices[itile].start - 1 # not necessary if A.rowindices == A.colindices
  invAjj = one(T)
  if xinvAjj
    @inbounds invAjj = 1 / tile(A, itile, jtile)[j - Δi, j - Δj]
  end
  @inbounds for itile in itiles
    A.isempties[itile, jtile] && continue
    tilerows = A.rowindices[itile]
    Δi = tilerows.start - 1
    t = tile(A, itile, jtile)
    if rows.start <= tilerows.start <= tilerows.stop <= rows.stop
      s = tileloop!(s, t, b, j - Δj, 1:size(t, 1), tilerows)
    else
      is = intersect(rows, tilerows)
      isempty(is) && continue
      s = tileloop!(s, t, b, j - Δj, is .- Δi, is)
      #s = s .+ transpose(view(b, is, :)) * view(t, is .- Δi, j - Δj)
    end
  end
  finalloop!(b, s, invAjj, j)
end
createsummand(b::AbstractMatrix) = zeros(eltype(b), size(b, 2))
createsummand(b::AbstractVector) = zero(eltype(b))
function LinearAlgebra.ldiv!(A::RLLU{T}, b::AbstractVecOrMat{T},
    transposeback=false) where {T}
  if !A.istransposed[]
    transpose!(A)
  end
  n = size(A, 1)
  # Solve Ly = b
  s = createsummand(b)
  for j in 2:n
    jtile = findtile(j, A.colindices)
    itilemax = findtile(j-1, A.rowindices)
    hotloopldiv!(s, A, b, 1:j-1, j, 1:itilemax, jtile, false)
  end
  # Solve Ux = y
  @views b[n, :] ./= A.A[n, n]
  for j in n-1:-1:1
    jtile = findtile(j, A.colindices)
    itilemin = findtile(j+1, A.rowindices)
    hotloopldiv!(s, A, b, j+1:n, j, itilemin:A.ntiles, jtile, true)
  end
  if A.istransposed[] && transposeback
    transpose!(A)
  end
  return b
end

end # module

