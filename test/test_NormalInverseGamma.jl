module TestDistributions
using Test
using Distributions
using EvidentialDeepLearning
using SpecialFunctions: loggamma

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

function S26(yᵢ,γ,ν,α,β)
    # taken from [Deep Evidential Regression](https://arxiv.org/abs/1910.02600)
    Ω = 2*β*(1+ν)
    1//2*log(π/ν) - α*log(Ω) + (α + 1//2) * log((yᵢ-γ)^2*ν+Ω) + loggamma(α) - loggamma(α+1//2)
end

@testset "Deep Evidential Regression (S26)" begin
    for _ in 1:100
        yᵢ = rand(Uniform(-10,10))
        γ = rand(Uniform(-5,5))     # Normal mean
        ν = rand(Uniform(0,100))    # nobs mean
        α = rand(Uniform(1, 50))    # nobs variance
        β = rand(Uniform(0,10)) * α # halfsum of square deviations

        nig = NIG(γ,ν,α,β)
        pp = posterior_predictive(nig)
        @test nll(pp, yᵢ) ≈ S26(yᵢ,γ,ν,α,β)
    end
end





end#module
