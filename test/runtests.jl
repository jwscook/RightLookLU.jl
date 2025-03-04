using LinearAlgebra, SparseArrays
using Random; Random.seed!(0)
using ThreadPinning
pinthreads(:cores)

using Test
using RightLookLU
using Chairmarks, BenchmarkTools
using SparseMatricesCSR

function test_matrix(T=Float64; ntiles=6, tilesize=5, overlap=0)
  n = ntiles * tilesize - overlap * (ntiles - 1)
  mat = zeros(T, n,n)
  for i in 1:ntiles
    j = (i - 1) * tilesize + 1 - overlap * max(0, i - 1)
    mat[j:j + tilesize - 1, j:j + tilesize - 1] .= rand(T, tilesize, tilesize)
  end
  mat .+= sprand(T, n, n, 0.1)
  return sparse(mat)
end

function mybenchmark(f::F, args...; n=NBENCH) where F
  return minimum([(newargs = deepcopy.(args); @elapsed f(newargs...)) for _ in 1:n])
end

const NBENCH=8

using StatProfilerHTML, Profile
function foo()
   ntiles=8
   tilesize=128
  s = test_matrix(ComplexF64; ntiles=ntiles, tilesize=tilesize, overlap=tilesize ÷ 8)
  s .+= 10 * I(size(s, 1))
  s2 = deepcopy(s)
  s2.nzval .= rand(ComplexF64, length(s2.nzval)) # == sparsity pattern, != values
  b = rand(ComplexF64, size(s, 1))
  x = s \ b
  sc = deepcopy(s)
  rl = RLLUMatrix(sc, min(16, size(sc, 1)))
  lurl = lu!(rl)
  xrl = deepcopy(b)
  ldiv!(rl, xrl)#xrl = rl \ b
  @test xrl ≈ x
  Profile.init(n=10^7, delay=1e-6)
  Profile.clear()
  ldiv!(x, lurl, b)
  @profilehtml ldiv!(x, lurl, b)
  run(`rm -rf ldiv3arg`)
  run(`mv -f statprof ldiv3arg`)
  lu!(lurl, s2)
  ldiv!(lurl, b)
  Profile.clear()
  @profilehtml ldiv!(lurl, b)
  run(`rm -rf ldiv2arg`)
  run(`mv -f statprof ldiv2arg`)
#  @profilehtml lu!(rl, s2)
  #return

@testset "rightlooklu!" begin
  for ntiles in (16,32), tilesize in (32,64,128), overlap in (tilesize ÷ 8,)
    s = test_matrix(Float64; ntiles=ntiles, tilesize=tilesize, overlap=overlap)
    s .+= 10 * I(size(s, 1))
    sc = deepcopy(s)
    lu_s = lu(deepcopy(s))
#    tb1 = (@benchmark lu!($s, NoPivot())).times |> minimum
    tb1 = (@b lu!($s, NoPivot())).time
#    tb2 = (@benchmark lu!($lu_s, $sc)).times |> minimum
    tb2 = (@b lu!($lu_s, $sc)).time
    b = rand(Float64, size(s, 1))
#    tb3 = (@benchmark $lu_s \ $b).times |> minimum
    tb3 = (@b $lu_s \ $b).time
    for lutiles in unique(min.(size(s, 1), (16, 32, 64)))
      lutiles > size(s, 1) && continue
      sc = deepcopy(s)
      LR = RLLUMatrix(sc, lutiles)
      lulrlu = deepcopy(lu!(LR))
      L2, U2 = tril(LR.A, -1) .+ I(size(LR.A, 1)), triu(LR.A)
      #@show norm(L2 * U2 - s) / norm(s)
      @test L2 * U2 ≈ s
      #ta1 = (@benchmark lu!(RLLUMatrix($s, $lutiles))).times |> minimum
      ta1 = (@b lu!(RLLUMatrix($s, $lutiles))).time
      #ta2 = (@benchmark lu!($lulrlu, $s)).times |> minimum
      ta2 = (@b lu!($lulrlu, $s)).time
      transpose!(lulrlu)
      x = s \ b
      @test x ≈ lulrlu \ b
      #ta3 = (@benchmark $lulrlu \ $b).times |> minimum
      ta3 = (@b $lulrlu \ $b).time
      Lcsc, Ucsc = lu(sc, NoPivot())
#      Lmatrix[diagind(LMatrix)] .= 1
#      Umatrix = UpperTriangular(Matrix(lulrlu.A[:, :]))
      #Lcsr, Ucsr = SparseMatrixCSR(Lcsc), SparseMatrixCSR(Ucsc)
      ta4 = (@b ($Ucsc \ ($Lcsc \ $b))).time
      #ta5 = (@b ($Ucsr \ ($Lcsr \ $b))).time
      #ta6 = (@b ($Ucsr \ ($Lcsc \ $b))).time
      #ta7 = (@b ($Ucsc \ ($Lcsr \ $b))).time
      n = size(s, 1)
      @show n, ntiles, tilesize, overlap, lutiles
      @show ta1 / tb1, ta2 / tb2, ta3 / tb3, ta4 / tb3#, ta5 / tb3, ta6 / tb3, ta7 / tb3
    end
  end

end
end
foo()
