module EvidentialDeepLearning
import DistributionsAD

const REF_SKK2018 = "['Evidential Deep Learning to Quantify Classification Uncertainty'](https://arxiv.org/abs/1806.01768)"

const REF_ASSR2019 = "['Deep Evidential Regression'](https://arxiv.org/abs/1910.02600)"

sampletype(o) = sampletype(typeof(o))
function sampletype(T::Type)
    error("""
            sampletype(::Type{$T}) not implemented.
          """)

end

include("NormalInverseGamma.jl")
include("Normal.jl")
include("Dirichlet.jl")
include("losses.jl")


end
