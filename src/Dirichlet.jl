using SpecialFunctions: loggamma, digamma
using StaticArrays

export Dirichlet, DirichletClassification, DirichletMultinomial, Multinomial

"""
    Dirichlet{N,T}

Statically sized Dirichlet distributions. In contrast to say `Distributions.Dirichlet` the
the number of categories is encoded as a type parameter.
"""
struct Dirichlet{N,T}
    α::NTuple{N,T}
    Dirichlet(α::NTuple{N,T}) where{N,T} = new{N,T}(α)
end

Dirichlet(args...) = Dirichlet(promote(args...))
Dirichlet(t::Tuple) = Dirichlet(t...)
Dirichlet(d::Dirichlet) = d
function Base.show(io::IO, d::Dirichlet)
    print(io, "Dirichlet", d.α)
end

to_NTuple(::Val{N}, x::NTuple{N}) where {N} = x
to_NTuple(::Val{N}, x::StaticVector{N}) where {N} = Tuple(x)
to_NTuple(::Val{N}, itr) where {N} = NTuple{N}(itr)::NTuple{N}

function Distributions.logpdf(d::Dirichlet{N}, x) where {N}
    x = to_NTuple(Val(N), x)
    α = d.α
    A = sum(loggamma.(α)) - loggamma(sum(α))
    sum(log.(x) .* (α .- 1)) - A
end

"""

    DirichletClassification(Val(k))

Layer that takes a tensor of floats of size (k*n, b) and turns it into
a tensor of Dirichlet distributions of size (n,b).
"""
struct DirichletClassification{k}
    K::Val{k}
end

"""

    kl_uniform(d::Dirichlet)

Given `P = Dirichlet(α)` compute the Kullback Leibler divergence with respect to the uniform Dirichlet distribution:

    `D_KL(P || Dirichlet(1,1,1...))`
"""
function kl_uniform(d::Dirichlet)
    entropy_uniform = -loggamma(nclasses(d))
    entropy_uniform - entropy(d)
end

nclasses(d::Dirichlet{N}) where {N} = N
function Distributions.entropy(d::Dirichlet)
    k = nclasses(d)
    α = d.α
    α₀ = sum(α)
    ψ = digamma
    logBα = log_multivariate_beta(α)
    logBα + (α₀ - k)* ψ(α₀) - sum((α .- 1) .* ψ.(α))
end

function log_multivariate_beta(α)
    sum(loggamma, α) - loggamma(sum(α))
end

function Statistics.mean(d::Dirichlet)
    a = inv(sum(d.α))
    return a * d.α
end

struct DirichletMultinomial{N,T}
    α::NTuple{N,T}
end
function predictive_posterior(d::Dirichlet)
    DirichletMultinomial(d.α)
end

struct Multinomial{N,T}
    p::NTuple{N,T} # p1...p1 Σpi = 1
end

# conversion
Distributions.Dirichlet(d::Dirichlet) = Distributions.Dirichlet(collect(d.α))
Distributions.Multinomial(d::Multinomial) = Distributions.Multinomial(d.p)
Distributions.DirichletMultinomial(d::DirichletMultinomial) = Distributions.DirichletMultinomial(d.α)

for (D_EDL, D_Dist) in [
                        (:Dirichlet, :(Distributions.Dirichlet)),
                        (:DirichletMultinomial, :(Distributions.DirichletMultinomial)),
                        (:Multinomial, :(Distributions.Multinomial)),
                        ]

    @eval Base.convert(::Type{Distributions.Distribution}, d::$D_EDL) = Base.convert($D_Dist, d)
    @eval Base.convert(::Type{$D_Dist}, d::$D_EDL) = $D_Dist(d)
end
