module RightLookLU

using Base.Threads, LinearAlgebra, SparseArrays
using ChunkSplitters
using BlockArrays
using TriangularSolve
#using HybridBlockArrays
#import HybridBlockArrays: tile

export RLLUMatrix

#lsolve!(A, L::LowerTriangular{<:Number, <:AbstractSparseMatrix}, _) = (A .= L \ A) # can't mutate L
function lsolve!(A, L::LowerTriangular{<:Number, <:AbstractSparseMatrix}, _)
  for j in 1:size(A, 2)
    vAj = view(A, :, j)
    iszero(vAj) || (vAj .= L \ vAj)
  end
end
lsolve!(A, L, work) = ldiv!(A, L, A)
#rsolve!(A, U::UpperTriangular{<:Number, <:AbstractSparseMatrix}, _) = (A .= A / U) # can't mutate U
rsolve!(A, U, work) = rdiv!(A, U)
function rsolve!(A, U::UpperTriangular{<:Number, <:AbstractSparseMatrix}, _)
  for i in 1:size(A, 1)
    vAi = transpose(view(A, i, :))
    iszero(vAi) || (vAi .= vAi / U)
  end
end

struct RLLUMatrix{T, M<:AbstractMatrix{T}}
  A::M
  ntiles::Int
  rowindices::Vector{UnitRange{Int64}}
  colindices::Vector{UnitRange{Int64}}
  isempties::Matrix{Bool}
  works::Vector{Matrix{T}}
  istransposed::Ref{Bool}
end
function RLLUMatrix(A::AbstractMatrix, ntiles::Int=32)
  @assert ntiles <= minimum(size(A))
  rowindices = collect(chunks(1:size(A, 1); n=ntiles))
  colindices = collect(chunks(1:size(A, 2); n=ntiles))
  A = BlockArray(A, [length(is) for is in rowindices], [length(js) for js in colindices])
  #A = HybridBlockArray(A, rowindices, colindices; sparsitythreshold=0.2)
  isempties = zeros(Bool, length(rowindices), length(colindices))
  for i in eachindex(rowindices), j in eachindex(colindices)
    isempties[i, j] = iszero(tile(A, i, j))
  end
  works = [similar(A, maximum(length(is) for is in rowindices),
                      maximum(length(js) for js in colindices)) for _ in 1:nthreads()]
  return RLLUMatrix(A, ntiles, rowindices, colindices, isempties, works, Ref(false))
end
Base.size(A::RLLUMatrix) = size(A.A)
Base.size(A::RLLUMatrix, i) = size(A.A, i)

tile(A::RLLUMatrix{T}, i, j) where {T} = tile(A.A, i, j)
tile(A::BlockArray{T}, i, j) where {T} = blocks(A)[i, j]

respectsparsitycopyto!(a, b) = copyto!(a, b)
function respectsparsitycopyto!(a::T, b::T) where {T<:AbstractSparseMatrix}
  a .= 0
  s[findall(!iszero, b)] .+= b.nzvals
end

struct RLLUStruct{T, M<:AbstractMatrix{T}}
  A::RLLUMatrix{T, M}
end

function LinearAlgebra.lu!(RL::RLLUMatrix)
  for level in 1:RL.ntiles
    factorise!(RL, level)
  end
  return RLLUStruct(RL)
end

function LinearAlgebra.lu!(RLS::RLLUStruct, A::AbstractMatrix)
  RL = RLS.A
  tasks = Task[]
  if RL.istransposed[]
    RL.istransposed[] = false
    RL.isempties .= RL.isempties'
  end
  for (i, is) in enumerate(RL.rowindices), (j, js) in enumerate(RL.colindices)
    RL.isempties[i, j] && continue # must have same sparsity pattern TODO check
    push!(tasks, @spawn respectsparsitycopyto!(tile(RL, i, j), view(A, is, js)))
  end
  wait.(tasks)
  #@assert all(RL.A .≈ A)
  lu!(RL)
  return RLS
end
LinearAlgebra.transpose!(A::RLLUStruct) = transpose!(A.A)
function LinearAlgebra.transpose!(A::RLLUMatrix)
  tmp = deepcopy(A.rowindices)
  A.rowindices .= A.colindices
  A.colindices .= tmp
  transpose!(A.A)
  A.isempties .= A.isempties'
  A.istransposed[] = !A.istransposed[]
  return A
end
function LinearAlgebra.transpose!(A::RLLUMatrix{T, <:BlockArray}) where T
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
function factorise!(A::RLLUMatrix, level)
  All = subtractleft!(A, level, level)
  Lll, Ull = diagonallu!(All, A.works[threadid()])
  Lll = LowerTriangular(Lll) # non-allocating
  Ull = UpperTriangular(Ull) # non-allocating
  tasks = Task[]
  for k in level + 1:A.ntiles
    push!(tasks, @spawn lookright!(A, level, k, Lll, Ull))
    push!(tasks, @spawn lookright!(A, k, level, Lll, Ull))
  end
  wait.(tasks)
  return A
end

function lookright!(A::RLLUMatrix, i, j, L, U)
  @assert i != j
  Aij = subtractleft!(A, i, j)
  A.isempties[i, j] && return
  i > j && rsolve!(Aij, U, A.works[threadid()]) # left hand columns, col-major op
  i < j && lsolve!(Aij, L, A.works[threadid()]) # right hand rows, row-major op
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

function subtractleft!(A::RLLUMatrix, i, j)
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

fastin(i, inds::UnitRange) = inds.start <= i <= inds.stop
function findtile(i::Int, indices)::Int
  for (ii, inds) in enumerate(indices)
    fastin(i, inds) && return ii
  end
  throw(BoundsError("$i not in $indices"))
  return 0
end
function tileloop!(s::Number, t::SparseMatrixCSC, b, j, trows::U, brows::U) where {U<:UnitRange}
  cA, cZ = t.colptr[j], (t.colptr[j+1] - 1)
  tA = trows.start
  tZ = trows.stop
  vt = view(t.rowval, cA:cZ)
  cZ = searchsortedlast(vt, tZ) + cA - 1
  cZ == 0 && return s
  cA = searchsortedfirst(vt, tA) + cA - 1
  (cA > cZ) && return s
  Δi = brows.start - tA
  @inbounds @simd for c in cA:cZ
    i = t.rowval[c]
    tc = t.nzval[c]
    bi = b[i + Δi]
    s = muladd(tc, bi, s)
    #s += t.nzval[c] * b[t.rowval[c] + Δi]
  end
  return s
end
function tileloop!(s, t::SparseMatrixCSC, b, j, trows::U, brows::U) where {U<:UnitRange}
  cA, cZ = t.colptr[j], (t.colptr[j+1] - 1)
  tA = trows.start
  tZ = trows.stop
  vt = view(t.rowval, cA:cZ)
  cZ = searchsortedlast(vt, tZ) + cA - 1
  cZ == 0 && return s
  cA = searchsortedfirst(vt, tA) + cA - 1
  (cA > cZ) && return s
  Δi = brows.start - tA
  @inbounds for c in cA:cZ
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
  return s + sum(i->tv[i] * bv[i], eachindex(tv))#BLAS.dotu(tv, bv)
end
function tileloop!(s, t, b, j, trows, brows)
  tv = view(t, trows, j)
  bv = view(b, brows, :)
  # gemm!(tA, tB, a, A, B, b, C) # C = a*A*B + b*C
#  BLAS.gemm!('T', 'N', one(eltype(s)), bv, tv, one(eltype(s)), s)
#  return s
  return s + transpose(b) * t
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

function hotloopldiv!(s, A::RLLUMatrix{T}, b, rows, j, itiles, jtile, xinvAjj::Bool) where {T}
  s = prepsummand(s)
  all(A.isempties[itile, jtile] for itile in itiles) && return
  #for i in rows, k in 1:size(b, 2); s[k] += A.A[i, j] * b[i, k]; end; return ### the default
  Δj = A.colindices[jtile].start - 1
  itile = findtile(j, A.rowindices) # not necessary if A.rowindices == A.colindices
  Δi = A.rowindices[itile].start - 1 # not necessary if A.rowindices == A.colindices
  invAjj = one(T)
  if xinvAjj
    @inbounds invAjj = 1 / tile(A, itile, jtile)[j - Δi, j - Δj]
  end
  @inbounds for itile in itiles
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

LinearAlgebra.:\(A::RLLUStruct, b) = ldiv!(A.A, deepcopy(b))
LinearAlgebra.:\(A::RLLUMatrix, b) = ldiv!(A, deepcopy(b))
function LinearAlgebra.ldiv!(x, A::RLLUStruct, b)
  x .= b
  return ldiv!(A.A, x)
end
LinearAlgebra.ldiv!(A::RLLUStruct, b) = ldiv!(A.A, b)
function LinearAlgebra.ldiv!(A::RLLUMatrix{T}, b::AbstractVecOrMat{T},
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

