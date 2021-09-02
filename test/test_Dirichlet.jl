module TestDirichlet
using Test
using EvidentialDeepLearning
const EDL = EvidentialDeepLearning
import Distributions; const DI = Distributions
using Distributions: logpdf, entropy
using SpecialFunctions: loggamma

@testset "Against Distributions" begin
    for k in 1:10
        D111 = Dirichlet(Tuple(1 for _ in 1:k))
        @test EDL.nclasses(D111) === k
        p = rand(k); p = p / sum(p)
        @test logpdf(D111, p) ≈ loggamma(k)
        @test entropy(D111) ≈ -loggamma(k)

        @test EDL.kl_uniform(D111) ≈ 0 atol=100eps(Float64)
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
        if EDL.nclasses(d_edl) == 1
            @test EDL.kl_uniform(d_edl) == 0
        else
            @test EDL.kl_uniform(d_edl) > 0
        end
    end
end

end#module
