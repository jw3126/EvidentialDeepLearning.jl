using Statistics: mean
using Distributions: logpdf
using EvidentialDeepLearning: evidence, predict, std_aleatoric
using LinearAlgebra: norm

export nll

"""

    nll(d,y;agg=mean)

Negative log likelihood loss. Given by `agg(-logpdf.(d,y))`.

# Arguments
* d (array of) distributions
* y (array of) points in the distributions domain.
"""
function nll(d,y;agg=mean)
    agg(-logpdf.(d,y))
end

"""

    evidence_regularizer(d,y;agg=mean)

Loss function for NormalInverseGamma regression.
Described in $(REF_ASSR2019).
"""
function evidence_regularizer(d,y;agg=mean)
    function evi_reg(d,y)
        err = y-predict(d)
        evidence(d) * norm(err)
    end
    agg(evi_reg.(d, y))
end

function regression_loss(d, y; λ=1f-2)
    pp = posterior_predictive.(d)
    nll(pp, y) + λ*evidence_regularizer(d,y)
end

"""

    dirichlet_sos(d, y; agg=mean, λ)

Regularized sum of squares loss for dirichlet classification. Described in $(REF_SKK2018).
"""
function dirichlet_sos(ds, labels::AbstractVector; λ)
    @argcheck eltype(ds) <: Dirichlet
    @argcheck size(ds) == size(labels)
    T = sampletype(eltype(ds))
    ret = float(zero(T))
    for (d, label) in zip(ds, labels)
        # https://github.com/aamini/evidential-deep-learning/blob/d1d8e395fb083308d14fa92c5ce766e97b2a066a/evidential_deep_learning/losses/discrete.py#L28
        d̂ = Dirichlet(setindex(d.α, one(T), label))
        ret += sos(d,label) + λ * kl_uniform(d̂)
    end
    return ret
end

function sos(d::Dirichlet{N}, label::Integer) where {N}
    α = d.α
    S = sum(α)
    m = α./S
    mse = (m[label] - 1)^2 - m[label]^2 + sum(abs2, m)
    mse + sum(α .* (S .- α) ./ (S .^2 .* S .+ 1))
end
