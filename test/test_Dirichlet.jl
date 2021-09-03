module TestDirichlet
using Test
using EvidentialDeepLearning
const EDL = EvidentialDeepLearning
import Distributions; const DI = Distributions
using Distributions: logpdf, entropy, Uniform
using SpecialFunctions: loggamma, digamma

@testset "Against Distributions" begin
    for k in 1:10
        D111 = Dirichlet(Tuple(1 for _ in 1:k))
        @test EDL.nclasses(D111) === k
        p = rand(k); p = p / sum(p)
        @test logpdf(D111, p) ≈ loggamma(k)
        @test entropy(D111) ≈ -loggamma(k)

        @test EDL.kl_uniform(D111) ≈ 0 atol=100eps(Float64)
    end

    for i in 1:10
        k = rand(1:10)
        d_edl  = Dirichlet(Tuple(10*rand(k)))
        d_dist = convert(DI.Distribution, d_edl)
        @test d_dist isa DI.Dirichlet
        p = rand(k)
        p = p / sum(p)
        @test logpdf(d_dist, p) ≈ logpdf(d_edl, p)
        @test entropy(d_dist) ≈ entropy(d_edl)
        if EDL.nclasses(d_edl) == 1
            @test EDL.kl_uniform(d_edl) == 0
        else
            @test EDL.kl_uniform(d_edl) > 0
        end
    end
end

@testset "against evidential-deep-learning" begin
    function _Dirichlet_SOS_KL_devec(d::EDL.Dirichlet)
        # Line by line port of
        # https://github.com/aamini/evidential-deep-learning/blob/d1d8e395fb083308d14fa92c5ce766e97b2a066a/evidential_deep_learning/losses/discrete.py#L6
        #
        α = d.α
        # beta=tf.constant(np.ones((1,alpha.shape[1])),dtype=tf.float32)
        β   = one.(α)
        # S_alpha = tf.reduce_sum(alpha,axis=1,keepdims=True)
        S_α = sum(α)
        # S_beta = tf.reduce_sum(beta,axis=1,keepdims=True)
        S_β = sum(β)
        # lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),axis=1,keepdims=True)
        lnB = loggamma(S_α) - sum(loggamma, α)
        # lnB_uni = tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(S_beta)
        # lnB_uni = tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(S_beta)
        lnB_uni = sum(loggamma, β) - loggamma(S_β)
        # dg0 = tf.math.digamma(S_alpha)
        dg0 = digamma(S_α)
        # dg1 = tf.math.digamma(alpha)
        dg1 = digamma.(α)
        # kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
        kl = sum((α .- β).*(dg1 .- dg0)) + lnB + lnB_uni
        # return kl
        return kl
    end
    for _ in 1:100
        k = rand(1:10)
        α = Tuple(rand(Uniform(0,19), k))
        d = Dirichlet(α)
        @test _Dirichlet_SOS_KL_devec(d) ≈ EDL.kl_uniform(d)
    end

end

end#module
