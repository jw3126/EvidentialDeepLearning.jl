module TestDistributions
using Test
using Distributions
using EvidentialDeepLearning

@testset "NormalInverseGamma" begin
    @testset "logpdf" begin
        for _ in 1:100
            μ = rand(Normal(0,10))
            ν = rand(Uniform(1,10))
            α = rand(Uniform(1,10))
            β = rand(Uniform(1,10))
            nig = NIG(μ,ν,α,β)
            x = rand(Normal(μ, 5))
            σ² = rand(Uniform(0,10))
            # We have
            # x |σ²,μ,ν ~ Normal(μ, √(σ²/ν)), x)
            # σ²|α,β    ~ InverseGamma(α,β)
            @test logpdf(nig, (;x,σ²)) ≈
                logpdf(Normal(μ, √(σ²/ν)), x) + logpdf(InverseGamma(α,β), σ²)
        end
    end
end





end#module
