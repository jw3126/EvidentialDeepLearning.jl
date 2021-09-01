module Losses
using Statistics: mean
using Distributions: logpdf

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

function evidence_regularizer(d,y;agg=mean)
    agg(evidence.(d) .* norm.(y .- predict.(d)))
end

end#module
