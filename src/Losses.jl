module Losses
using Statistics
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

end#module
