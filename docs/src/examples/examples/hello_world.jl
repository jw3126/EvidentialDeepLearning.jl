# # Hello World
# This tutorial is a port of [hello_world.py](https://github.com/aamini/evidential-deep-learning/blob/d1d8e395fb083308d14fa92c5ce766e97b2a066a/hello_world.py) to julia.
# We create a synthetic dataset, train a simple evidential regression model and plot the results
using Flux, Plots
import EvidentialDeepLearning as EDL
using Distributions
# Lets create some data
function my_data(x_min, x_max, n; train=true)
    x = range(Float32(x_min), Float32(x_max), length=n)
    x = reshape(x, (1,n))
    σ = train ? 3f0 : 0f0
    y = x .^3 .+ rand(Normal(0f0, σ), size(x))
    return x, y
end
x_train, y_train = my_data(-4, 4, 1000)
x_test, y_test = my_data(-7, 7, 1000, train=false)
plot(title="Dataset", ylims=(-150,150), xlims=(-7,7))
scatter!(vec(x_train), vec(y_train), label="train", markersize=1)
plot!(vec(x_test), vec(y_test), label="truth")

# Our model is a single hidden layer dense network, where the output is converted to NormalInverseGamma
# distributions
model = Flux.Chain(
    Dense(1, 64, Flux.relu),
    Dense(64, 4, identity),
    EDL.NIGs_from_4channels
)

# At the end of the day, the output of our model `d = model(x)` encodes three interesting informations:
# * A prediction of y, given by `mean(posterior_predictive(x))`
# * An uncertainty estimate, given by `std(posterior_predictive(x))`
# * The evidence `evidence(d)`, which encodes how certain the model is about the prediction.
# In order to optimize the model, we need a loss function, that covers these. The loss consists of two parts:
# * The first part is the negative log likelyhood of the data under the predictive posterior.
# * The regulizer part penalizes the net for outputting high evidence in in presence of wrong predictions
function loss(x, y)
    d = model(x)
    λ = 1f-2
    EDL.Losses.nll(EDL.posterior_predictive.(d), y) + λ*EDL.Losses.evidence_regularizer(d,y)
end
# Lets train the model
data = Flux.Data.DataLoader((x_train, y_train), batchsize=100, shuffle=true)
opt = ADAM(5e-4) # training is quite sensitive to learning rate
for epoch in 1:500 # takes ~ 30s
    Flux.train!(loss, Flux.params(model), data, opt)
end
# Lets plot the results.
plot(legend=:bottom)
h = model(x_test)
preds = vec(EDL.predict.(h))
plot!(vec(x_test), preds, label="Prediction", color=:blue, xlims=(-7,7), ylims=(-150,150))
σ_pp = vec(EDL.std_predict.(h)) # std of the predictive posterior
lo = preds - σ_pp
hi = preds + σ_pp
plot!(vec(x_test), hi, fillrange=lo, label="Predicted error", color=:green, alpha=0.5)
scatter!(vec(x_train), vec(y_train), label="training data", markersize=1, color=:red)
plot!(vec(x_test), vec(y_test), label="truth", color=:black)
