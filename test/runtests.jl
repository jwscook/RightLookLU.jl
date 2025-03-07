using LinearAlgebra, SparseArrays
using Random; Random.seed!(0)
using ThreadPinning
pinthreads(:cores)

using Test
using RightLookLU
using BenchmarkTools

function test_matrix(T=Float64; ntiles=6, tilesize=5, overlap=0)
  n = ntiles * tilesize - overlap * (ntiles - 1)
  mat = zeros(T, n,n)
  for i in 1:ntiles
    j = (i - 1) * tilesize + 1 - overlap * max(0, i - 1)
    mat[j:j + tilesize - 1, j:j + tilesize - 1] .= rand(T, tilesize, tilesize)
  end
  mat .+= sprand(T, n, n, 0.1)
  return sparse(mat)
  #return mat
end

function mybenchmark(f::F, args...; n=3) where F
  return minimum([(newargs = deepcopy.(args); @elapsed f(newargs...)) for _ in 1:n])
end

using StatProfilerHTML, Profile
function foo()
  ntiles=64 ÷ 4
  tilesize=256 ÷ 4
  s = test_matrix(ComplexF64; ntiles=ntiles, tilesize=tilesize, overlap=tilesize ÷ 4)
  s .+= 10 * I(size(s, 1))
  b = rand(ComplexF64, size(s, 1))
  x = s \ b
  sc = deepcopy(s)
  rl = RLLUMatrix(sc, min(16, size(sc, 1)))
  lurl = lu!(rl)
  xrl = rl \ b
  @test xrl ≈ x
  Profile.init(n=10^7, delay=1e-6)
  Profile.clear()
  @profilehtml ldiv!(x, lurl, b)
  #@assert false
  s2 = test_matrix(ComplexF64; ntiles=ntiles, tilesize=tilesize, overlap=tilesize ÷ 4)
  s2 .+= 10 * I(size(s2, 1))
  s2 .+= sprand(ComplexF64, size(s2, 1), size(s2, 2), 0.1)
  lu!(rl, s2)
  ldiv!(rl, b)
  Profile.clear()
#  @profilehtml ldiv!(rl, b)
  @profilehtml lu!(rl, s2)

  #return

@testset "rightlooklu!" begin
  #for n in (128,256), lutiles in (8, 16, 32)
  #  lutiles > n && continue
  #  A = rand(ComplexF64, n, n)
  #  L, U = lu(A, NoPivot())
  #  Ac = deepcopy(A)
  #  LR = RLLUMatrix(Ac, lutiles)
  #  lu!(LR)
  #  L1, U1 = tril(LR.A, -1) .+ I(n), triu(LR.A)
  #  @test L1 * U1 ≈ A
  #  t1 = mybenchmark((x...)->lu!(RLLUMatrix(x...)), A, lutiles; n=3)
  #  t2 = mybenchmark(x->lu!(x, NoPivot()), A; n=3)
  #  @show n, lutiles, t1 / t2
  #  @benchmark lu!(RLLUMatrix($A, $lutiles))
  #end
  for ntiles in (4, 8, 16), tilesize in (16, 32, 64), overlap in (tilesize ÷ 4,)
    s = test_matrix(ComplexF64; ntiles=ntiles, tilesize=tilesize, overlap=overlap)
    s .+= 10 * I(size(s, 1))
    lu_s = lu(deepcopy(s))
    tb1 = mybenchmark(x->lu!(x, NoPivot()), s; n=2)
    tb2 = typeof(s) <: SparseMatrixCSC ? mybenchmark(x->lu!(lu_s, x), s; n=2) : NaN
    b = rand(ComplexF64, size(s, 1))
    tb3 = mybenchmark(x->x \ b, lu_s; n=2)
    for lutiles in unique(min.(size(s, 1), (4, 7, 8, 16, 17, 32, 64)))
      lutiles > size(s, 1) && continue
      sc = deepcopy(s)
      LR = RLLUMatrix(sc, lutiles)
      lulrlu = deepcopy(lu!(LR))
      L2, U2 = tril(LR.A, -1) .+ I(size(LR.A, 1)), triu(LR.A)
      #@show norm(L2 * U2 - s) / norm(s)
      @test L2 * U2 ≈ s
      ta1 = mybenchmark((x...)->lu!(RLLUMatrix(x...)), s, lutiles; n=3)
      ta2 = mybenchmark(x->lu!(LR, x), s; n=3)
  #    @show ntiles, tilesize, overlap, lutiles, ta1 / tb1, ta2 / tb2
      transpose!(lulrlu)
      x = s \ b
      @test x ≈ lulrlu \ b
      #@show norm(x - lulrlu \ b)
      ta3 = mybenchmark(x->x \ b, lulrlu; n=3)
      @show ntiles, tilesize, overlap, lutiles, ta1 / tb1, ta2 / tb2, ta3 / tb3
    end
  end

end
end
foo()
