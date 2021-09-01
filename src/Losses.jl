module Losses
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

This loss function penalizes values of `d` which have high evidence and high
prediction error with respect to y.
"""
function evidence_regularizer(d,y;agg=mean)
    function evi_reg(d,y)
        err = y-predict(d)
        evidence(d) * norm(err)
    end
    agg(evi_reg.(d, y))
end

end#module
