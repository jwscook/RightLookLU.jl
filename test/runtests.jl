using LinearAlgebra, SparseArrays
using Random; Random.seed!(0)
using ThreadPinning
pinthreads(:cores)

using Test
using RightLookLU

function test_matrix(T=Float64; ntiles=6, tilesize=5, overlap=1)
  n = ntiles * tilesize - (tilesize - overlap - 1) * (ntiles - 1)
  mat = zeros(T, n,n)
  for i in 1:ntiles
    j = (i - 1) * tilesize - (tilesize - overlap - 1) * (i - 1) + 1
    mat[j:j + tilesize - 1, j:j + tilesize - 1] .= rand(T, tilesize, tilesize)
  end
  return sparse(mat)
end

function mybenchmark(f::F, args...; n=3) where F
  return minimum([(newargs = deepcopy.(args); @elapsed f(newargs...)) for _ in 1:n])
end

using StatProfilerHTML
function foo()
  s = test_matrix(ComplexF64; ntiles=32, tilesize=1024, overlap=16)
  s .+= 10 * I(size(s, 1))
  b = rand(ComplexF64, size(s, 1))
  x = s \ b
  sc = deepcopy(Matrix(s))
  rl = RLLU(sc, 16)
  lurl = lu!(rl)
  xrl = rl \ b
  @test xrl ≈ x
  @profilehtml [ldiv!(x, lurl, b) for _ in 1:10]
  #@assert false
  s2 = test_matrix(ComplexF64; ntiles=32, tilesize=1024, overlap=16)
  s2 .+= 10 * I(size(s2, 1))
  lu!(rl, s2)
#  @profilehtml [lu!(rl, s2) for _ in 1:10]

  #return

@testset "rightlooklu!" begin
  for n in 2 .^ (4, 6, 8, 10), lutiles in (2, 4, 8, 16)
    A = rand(ComplexF64, n, n)
    L, U = lu(A, NoPivot())
    Ac = deepcopy(A)
    LR = RLLU(Ac, lutiles)
    lu!(LR)
    L1, U1 = tril(LR.A, -1) .+ I(n), triu(LR.A)
    @test L1 * U1 ≈ A
    t1 = mybenchmark((x...)->lu!(RLLU(x...)), A, lutiles; n=10)
    t2 = mybenchmark(x->lu!(x, NoPivot()), A; n=10)
    @show n, lutiles, t1 / t2
  end
  for ntiles in (8, 16), tilesize in (64, 128, 256, 512), overlap in (4,) 
    s = test_matrix(ComplexF64; ntiles=ntiles, tilesize=tilesize, overlap=overlap)
    s .+= 10 * I(size(s, 1))
    lu_s = lu(deepcopy(s))
    tb1 = mybenchmark(x->lu!(x, NoPivot()), s; n=20)
    tb2 = mybenchmark(x->lu!(lu_s, x), s; n=20)
    b = rand(ComplexF64, size(s, 1))
    tb3 = mybenchmark(x->x \ b, lu_s; n=20)
    for lutiles in unique(min.(size(s, 1), (8, 16, 32, 64)))
      sc = deepcopy(s)
      LR = RLLU(sc, lutiles)
      lulrlu = deepcopy(lu!(LR))
      L2, U2 = tril(LR.A, -1) .+ I(size(LR.A, 1)), triu(LR.A)
      @test L2 * U2 ≈ s
      ta1 = mybenchmark((x...)->lu!(RLLU(x...)), s, lutiles; n=20)
      ta2 = mybenchmark(x->lu!(LR, x), s; n=20)
  #    @show ntiles, tilesize, overlap, lutiles, ta1 / tb1, ta2 / tb2
      luumfpack = lu(s)
      ta3 = mybenchmark(x->x \ b, lulrlu; n=20)
      @show ntiles, tilesize, overlap, lutiles, ta1 / tb1, ta2 / tb2, ta3 / tb3
    end
  end

end
end
foo()
