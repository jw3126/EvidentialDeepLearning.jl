module TestDirichlet
using Test
using EvidentialDeepLearning
import Distributions; const DI = Distributions
using Distributions: logpdf, entropy
using SpecialFunctions: loggamma

@testset "Against Distributions" begin
    for k in 1:10
        @test entropy(Dirichlet(Tuple(1 for _ in 1:k))) ≈ -loggamma(k)
    end

    for i in 1:10
        k = rand(1:10)
        d_edl  = Dirichlet(Tuple(10*rand(k)))
        d_dist = convert(DI.Distribution, d_edl)
        @test d_dist isa DI.Dirichlet
        p = rand(k)
        p = p / sum(p)
        @test logpdf(d_dist, p) ≈ logpdf(d_edl, p)
        @test entropy(d_dist) ≈ entropy(d_edl)
    end
end

end#module
