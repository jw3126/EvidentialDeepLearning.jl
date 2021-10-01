struct Normal{T}
    μ::T
    σ::T
    function Normal(m,s)
        μ,σ = promote(m,s)
        T = typeof(μ)
        new{T}(μ, σ)
    end
end
Normal(;μ,σ) = Normal(μ,σ)
Base.show(io::IO, o::Normal) = print(io, "Normal(μ=$(o.μ), σ=$(o.σ))")

struct NormalRegressor end

function (o::NormalRegressor)(arr::AbstractMatrix)
    @argcheck mod(size(arr,1), 2) == 0
    nc, nb = size(arr)
    b = Int(nc/2)
    cmu    = view(arr, 1:b)
    csigma = view(arr, b+1:2b)
    return Normal_from_2channels.(cmu, csigma)
end
function Normal_from_2channels(cmu, csigma)
    return Normal(cmu, softplus(csigma))
end

function Distributions.logpdf(o::Normal, x)
    μ = o.μ
    σ = o.σ
    T = typeof(μ)
    half = T(1) / T(2)
    B = -1/(2*σ^2) * (μ - x)^2
    A = -half * log2π - log(σ)
    return A + B
end

sampletype(::Type{Normal{T}}) where {T} = T
Distributions.mean(o::Normal) = o.μ
Distributions.std(o::Normal)  = o.σ
Distributions.Normal(o::Normal) = Distributions.Normal(o.μ, o.σ)
