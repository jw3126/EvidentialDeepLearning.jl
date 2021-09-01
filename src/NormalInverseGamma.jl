using SpecialFunctions: loggamma
using ArgCheck
import Statistics
import Distributions
using StatsFuns: log2π
using ArgCheck: @argcheck
import Flux

################################################################################
##### NormalInverseGamma
################################################################################
export NIG
export posterior_predictive
export mean, var, logpdf

"""
    NIG(μ, ν, α, β)

[Normal-inverse-gamma distribution](https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution).
"""
struct NIG{T}
    "Sample mean."
    μ::T
    "Number of pseudo observations of the mean. Called λ in wikipedia."
    ν::T
    "Half number of pseudo observations for the square deviations."
    α::T
    "Half sum of square deviations from the mean."
    β::T # 2β = square errors
    function NIG(μ, ν, α, β)
        mu, nu, alpha, beta = promote(μ, ν, α, β)
        T = typeof(nu)
        new{T}(mu, nu, alpha, beta)
    end
end
NIG(;μ, ν, α, β) = NIG(μ, ν, α, β)

function Base.show(io::IO, nig::NIG)
    print(io, "NIG(μ=$(nig.μ), ν=$(nig.ν), α=$(nig.α), β=$(nig.β))")
end


function NIGs_from_4channels(arr::AbstractMatrix)
    nc,nb = size(arr)
    @argcheck nc === 4
    cmu    = view(arr, 1,:)
    cnu    = view(arr, 2,:)
    calpha = view(arr, 3,:)
    cbeta  = view(arr, 4,:)
    map(NIG_from_4channels, cmu, cnu, calpha, cbeta)
end

function NIG_from_4channels(cμ, cν, cα, cβ)
    μ = cμ
    ν = Flux.softplus(cν)
    α = Flux.softplus(cα) + 1
    β = Flux.softplus(cβ)
    NIG(μ,ν,α,β)
end

function to_nig_sampletype(pt)
    @argcheck length(pt) == 2
    x,σ²= pt
    return (;x,σ²)
end
function to_nig_sampletype(nt::NamedTuple)
    return (;nt.x, nt.σ²)
end

function Distributions.logpdf(nig::NIG, pt)
    (;x, σ²) = to_nig_sampletype(pt)
    μ = nig.μ
    λ = nig.ν
    α = nig.α
    β = nig.β
    l = -(2β + λ*(x-μ)^2) / (2*σ²)
    logσ² = log(σ²)
    #return √(λ)       / √(2π*σ²)          * β^α / SPF.gamma(α)       * inv(σ²)^(α+1) * exp(l)
    return 1/2*log(λ) - 1/2*(log2π +logσ²) + α*log(β) - loggamma(α) - logσ²*(α+1) + l
end

Statistics.mean(nig::NIG) = (x=nig.μ, σ²=nig.β/(nig.α - 1))
function Statistics.var(nig::NIG)
    μ = nig.μ
    λ = nig.ν
    α = nig.α
    β = nig.β
    Eσ² = mean(nig).σ²
    x  = Eσ²/ λ
    σ² =  Eσ² / (α-2)
    return (;x,σ²)
end

"""
    posterior_predictive(nig::NIG)

Return the posterior_predictive distribution, where the statistical
model is the two parameter family `Normal(μ,σ²)` and the posterior is given by `nig`.
"""
function posterior_predictive(nig::NIG)
    # https://en.wikipedia.org/wiki/Conjugate_prior
    μ = nig.μ # mean
    α = nig.α # 2α = nobs var
    ν = nig.ν # nobs mean
    β = nig.β # 2β = sum square dev
    var = β * (ν+1) / (ν*α)  # taken from Wikipedia
    studentνμσ(2α, μ, sqrt(var))
end

"""
    studentνμσ(ν,μ,σ)

Create a student t distributions with
* ν degrees of freedom
* μ location (= mean)
* σ scale (≈ std deviation)
"""
function studentνμσ(ν,μ,σ)
    d0 = TDist(ν)
    LocationScale(μ,σ,d0)
end
