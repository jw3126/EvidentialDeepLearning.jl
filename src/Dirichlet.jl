using SpecialFunctions: loggamma, digamma
import Distributions: entropy
using StaticArrays

export Dirichlet, DirichletClassifier, Categorical

################################################################################
##### Dirichlet
################################################################################
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

to_NTuple(::Val{n}, x::NTuple{n}) where {n} = x
to_NTuple(::Val{n}, x::StaticVector{n}) where {n} = Tuple(x)
to_NTuple(N::Val, itr) = to_NTuple_dynamic(N, itr)

to_NTuple_dynamic(::Val{0}, itr) = ()
to_NTuple_dynamic(::Val{1}, itr) = (first(itr),)
function to_NTuple_dynamic(::Val{2}, itr)
    x1,x2 = itr
end
function to_NTuple_dynamic(::Val{3}, itr)
    x1,x2,x3 = itr
end
function to_NTuple_dynamic(::Val{4}, itr)
    x1,x2,x3,x4 = itr
end
to_NTuple_dynamic(::Val{n}, itr) where {n} = NTuple{n}(itr)::NTuple{n}

function Distributions.logpdf(d::Dirichlet{N}, x) where {N}
    x = to_NTuple(Val(N), x)
    α = d.α
    A = sum(loggamma.(α)) - loggamma(sum(α))
    sum(log.(x) .* (α .- 1)) - A
end

"""

    DirichletClassifier(Val(k))

Layer that takes a tensor of floats of size (k*n, b) and turns it into
a tensor of Dirichlet distributions of size (n,b).
"""
struct DirichletClassifier{k}
    K::Val{k}
end


val(::Val{k}) where {k} = k
NClasses(::Type{DirichletClassifier{k}}) where {k} = Val(k)
NClasses(o) = NClasses(typeof(o))
function NClasses(T::Type)
    error("""
          NClasses(::Type{$T}) not implemented.
          """)
end
nclasses(o) = val(NClasses(o))

function (o::DirichletClassifier)(arr::AbstractMatrix)
    @argcheck size(arr, 1) == nclasses(o)
    K = NClasses(o)
    rows = ntuple(K) do i
        view(arr, i, :)
    end
    map(rows...) do args...
        Dirichlet(softplus.(args)...)
    end
end

"""

    kl_uniform(d::Dirichlet)

Given `P = Dirichlet(α)` compute the Kullback Leibler divergence with respect to the uniform Dirichlet distribution:

    `D_KL(P || Dirichlet(1,1,1...))`
"""
function kl_uniform(d::Dirichlet)
    # For any distribution P = p(x)dx
    # And a uniform distribution U = u*dx where u = 1/vol(Ω)
    # We have
    # D_KL(P||U) = ∫ log(u/p(x)) p(x) dx
    #            = ∫ log(u) p(x) dx - ∫log(p(x)) p(x) dx
    #            = entropy(U)       - entropy(P)
    #
    # This also makes sense intuitively:
    # D_KL(P||U) measures the extra price we pay for using the uniform coding scheme
    # on P distributed data.
    # In the uniform scheme, we pay always a flat rate of entropy(U) for any sample,
    # while with the optimal code we only pay entropy(P) on average per sample
    entropy_uniform = -loggamma(nclasses(d))
    entropy_uniform - entropy(d)
end

NClasses(::Type{<:Dirichlet{k}}) where {k} = Val(k)
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

predict(d::Dirichlet) = argmax(d.α)
evidence(d::Dirichlet) = sum(d.α) - nclasses(d) # is this correct?

################################################################################
##### Categorical
################################################################################
struct Categorical{N,T}
    p::NTuple{N,T} # Σpᵢ = 1 must hold
end

function posterior_predictive(o::Dirichlet)
    Categorical(o.α ./ sum(o.α))
end
Base.@propagate_inbounds Distributions.logpdf(o::Categorical, i) = log(o.p[i])
Base.@propagate_inbounds Distributions.pdf(o::Categorical, i)    = o.p[i]

# conversion
Distributions.Dirichlet(d::Dirichlet) = Distributions.Dirichlet(collect(d.α))
Distributions.Categorical(d::Categorical) = Distributions.Categorical(d.p)

for (D_EDL, D_Dist) in [
                        (:Dirichlet, :(Distributions.Dirichlet)),
                        (:Categorical, :(Distributions.Categorical)),
                        ]

    @eval Base.convert(::Type{Distributions.Distribution}, d::$D_EDL) = Base.convert($D_Dist, d)
    @eval Base.convert(::Type{$D_Dist}, d::$D_EDL) = $D_Dist(d)
end

sampletype(::Type{Dirichlet{N,T}}) where {N,T} = T
sampletype(::Type{Categorical{N,T}}) where {N,T} = T
