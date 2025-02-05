module RightLookLU

using Base.Threads, LinearAlgebra, SparseArrays
using ChunkSplitters
using BlockArrays

export RLLU

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
  works = [similar(A, maximum(length(is) for is in rowindices),
                      maximum(length(js) for js in colindices)) for _ in 1:nthreads()]
  return RLLU(A, ntiles, rowindices, colindices, isempties, works, Ref(false))
end
Base.size(A::RLLU) = size(A.A)
Base.size(A::RLLU, i) = size(A.A, i)

tile(A::RLLU{T, M}, i, j) where {T, M<:BlockArray{T}} = blocks(A.A)[i, j]

function tile(A::RLLU{T, M}, i, j) where {T, M}
  is, js = A.rowindices[i], A.colindices[j]
  return view(A.A, is, js)
end

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
  A.rowindices, A.colindices = A.colindices, A.rowindices
  transpose!(A.A)
  transpose!(A.empties)
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

function factorise!(A::RLLU, level)
  All = subtractleft!(A, level, level)
  Lll, Ull = lu!(All, NoPivot(); check=false)
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
function _subtractleftmul!(A::Matrix{T}, L, U) where T
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
  @show i, indices
  throw(BoundsError())
  return 0
end
function hotloopldiv!(s, A::RLLU{T}, b, rows, j, itiles, jtile) where {T}
  fill!(s, 0)
  #for i in rows, k in 1:size(b, 2)
  #  s[k] += A.A[i, j] * b[i, k]
  #end
  #return
  Δj = A.colindices[jtile].start - 1
  for itile in itiles
    A.isempties[itile, jtile] && continue
    tilerows = A.rowindices[itile]
    Δi = tilerows.start - 1
    t = tile(A, itile, jtile)
    if rows.start <= tilerows.start <= tilerows.stop <= rows.stop
        for k in 1:size(b, 2)
          @simd for i in axes(t, 1)
            s[k] += t[i, j - Δj] * b[i + Δi, k]
          end
        end
    else
      for i in intersect(rows, tilerows)
        for k in 1:size(b, 2)
          s[k] += t[i - Δi, j - Δj] * b[i, k]
        end
      end
    end
  end
end
function LinearAlgebra.ldiv!(A::RLLU{T}, b::AbstractVecOrMat{T},
    transposeback=false) where {T}
  # L won't have a unit diagonal so make that explicit in the loop
  if !A.istransposed[]
    transpose!(A)
  end
  n = size(A, 1)
  # Solve Ly = b
  s = zeros(T, size(b, 2))
  for j in 2:n
    jtile = findtile(j, A.colindices)
    itilemax = findtile(j-1, A.rowindices)
    hotloopldiv!(s, A, b, 1:j-1, j, 1:itilemax, jtile)
    @views b[j, :] .= (b[j, :] .- s)
  end
  # Solve Ux = y
  @views b[n, :] ./= A.A[n, n]
  for j in n-1:-1:1
    jtile = findtile(j, A.colindices)
    itilemin = findtile(j+1, A.rowindices)
    hotloopldiv!(s, A, b, j+1:n, j, itilemin:A.ntiles, jtile)
    @views b[j, :] .= (b[j, :] .- s) ./ A.A[j, j]
  end
  if A.istransposed[] && transposeback
    transpose!(A)
  end
  return b
end
#
#function LinearAlgebra.ldiv!(A::RLLU{T}, b::AbstractVecOrMat{T}) where T
#  #  L won't have a unit diagonal so make that explicit in the loop
#  n = size(A, 1)
#  # Solve Ly = b
#  @inbounds for i in 1:n
#    for k in 1:size(b, 2)
#      s = zero(T)
#      @simd for j in 1:i-1#
#        s += A.A[i, j] * b[j, k]
#      end
#      b[i, k] = (b[i, k] - s)
#    end
#  end
#  # Solve Ux = y
#  @inbounds for i in n:-1:1
#    for k in 1:size(b, 2)
#      s = zero(T)
#      @simd for j in i+1:n#
#        s += A.A[i, j] .* b[j, k]
#      end
#      b[i, k] = (b[i, k] .- s) / A.A[i,i]
#    end
#  end
#  return b
#end
#function LinearAlgebra.ldiv!(A::RLLU{T}, b::AbstractVecOrMat{T};
#        transposeback=false) where T
#  if !A.istransposed[]
#    transpose!(A)
#  end
#  #  L won't have a unit diagonal so make that explicit in the loop
#  n = size(A, 1)
#  # Solve Ly = b
#  @inbounds for j in 1:n
#    for k in 1:size(b, 2)
#      s = zero(T)
#      @simd for i in 1:j-1#
#        s += A.A[i,j] * b[i, k]
#      end
#      b[j, k] = (b[j, k] - s)# / L[i, i] # L[i,i] shares a diagonal with U
#    end
#  end
#  # Solve Ux = y
#  @inbounds for j in n:-1:1
#    for k in 1:size(b, 2)
#      s = zero(T)
#      @simd for i in j+1:n#
#        s += A.A[i,j] .* b[i, k]
#      end
#      b[j, k] = (b[j, k] - s) / A.A[j,j]
#    end
#  end
#  if A.istransposed[] && transposeback
#    transpose!(A)
#  end
#  return b
#end



end # module

