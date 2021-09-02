using SpecialFunctions: loggamma
using ArgCheck
import Statistics
import Distributions
using StatsFuns: log2π
using ArgCheck: @argcheck
using NNlib: softplus

################################################################################
##### NormalInverseGamma
################################################################################
export NIG
export NIGRegression
export posterior_predictive
export mean, var, logpdf

"""
    NIG(μ, ν, α, β)

[Normal-inverse-gamma distribution](https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution).
"""
struct NIG{T}
    "Sample mean."
    μ::T
    "Number of pseudo observations used to estimate μ. Wikipedia uses λ instead of ν."
    ν::T
    "Half number of pseudo observations used to estimate β."
    α::T
    "Half sum of square deviations from the mean."
    β::T
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

"""
NormalInverseGamma regression layer. Takes a tensor of floats of size (4c, n) and
outputs a tensor of NormalInverseGamma distributions of size (c,n).
"""
struct NIGRegression end

(o::NIGRegression)(x) = NIGs_from_4channels(x)

function NIGs_from_4channels(arr::AbstractMatrix)
    @argcheck mod(size(arr,1), 4) == 0
    nc,nb = size(arr)
    b = nc ÷ 4
    cmu    = view(arr, (   1):(1b),:)
    cnu    = view(arr, (1b+1):(2b),:)
    calpha = view(arr, (2b+1):(3b),:)
    cbeta  = view(arr, (3b+1):(4b),:)
    map(NIG_from_4channels, cmu, cnu, calpha, cbeta)
end

function NIG_from_4channels(cμ, cν, cα, cβ)
    μ = cμ
    ν = softplus(cν)
    α = softplus(cα) + 1
    β = softplus(cβ)
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
    s = to_nig_sampletype(pt)
    x  = s.x
    σ² = s.σ²
    μ  = nig.μ
    λ  = nig.ν
    α  = nig.α
    β  = nig.β
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
    d0 = Distributions.TDist(ν)
    Distributions.LocationScale(μ,σ,d0)
end

function evidence(o::NIG)
    2*o.ν + o.α
end
function predict(o::NIG)
    # This is E[μ]
    o.μ
end

"""
    var_aleatoric(d)

[Alleatoric](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic_uncertainty) variance of `predict(d)`. Also known as statistical uncertainty.

See also [var_epistemic](@ref).
"""
function var_aleatoric end
function var_aleatoric(o::NIG)
    # E(σ²)
    return o.β / (o.α - 1)
end

"""
    std_aleatoric(o) = √(var_aleatoric(o))
"""
std_aleatoric(o) = √(var_aleatoric(o))

"""
    var_epistemic(d)

[Epistemic](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic_uncertainty) variance of `predict(d)`. Also known as systematic uncertainty.

See also [var_aleatoric](@ref).
"""
function var_epistemic end

function var_epistemic(o::NIG)
    # Var(μ)
    o.β / ((o.α - 1) * o.ν)
end

"""
    std_epistemic(o) = √(var_epistemic(o))
"""
std_epistemic(o) = √(var_epistemic(o))

std_predict(o::NIG) = Statistics.std(posterior_predictive(o))
var_predict(o::NIG) = var(posterior_predictive(o))
