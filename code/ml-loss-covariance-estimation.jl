### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ f8bb97c3-a6be-400f-8d02-548843b44aa7
begin
	import Pkg
	Pkg.add("Distributions")
	Pkg.add("Zygote")
	Pkg.add("Flux")
	Pkg.add("DataFrames")
	Pkg.add("Plots")
	Pkg.add("LaTeXStrings")
	Pkg.add("ProgressLogging")
	Pkg.add("LinearRegression")
	Pkg.add("Optim")

	using Zygote: Zygote
	using Flux:Flux
	using MLDatasets: MNIST
	using DataFrames: DataFrame, push!, first
	using PlutoUI: Slider
	using LinearAlgebra: norm, LinearAlgebra
	using Statistics:Statistics
	using Distributions: Distributions
	using Plots: plot, plot!
	using LaTeXStrings
	using ProgressLogging
	using LinearRegression
	using Optim
end

# ╔═╡ 9eb1ddaa-9028-444e-8f35-218ba57d1e99
md"# MNIST Dataset, model and loss

We want to train a standard model (taken from Github) on the MNIST dataset using the
mean squared loss
"

# ╔═╡ 677320e3-d796-404b-9d6b-f2e77ee947e2
begin
	x_train, y_train = MNIST(split=:train)[:]
	x_train = Float32.(x_train)
	y_train_oh = Flux.onehotbatch(y_train, 0:9)
	size(x_train), size(y_train), size(y_train_oh)
end

# ╔═╡ 11dcf7ce-2c57-11ee-0f5b-014ecac26614
## from https://github.com/ansh941/MnistSimpleCNN/blob/master/code/models/modelM7.py
function mnistSimpleCNN7()
	return Flux.Chain(
	    Flux.unsqueeze(3),
	    Flux.Conv((7,7), 1=>48, bias=false), # output becomes 22x22
	    Flux.BatchNorm(48, Flux.relu),
	    Flux.Conv((7,7), 48=>96, bias=false), # output becomes 16x16
	    Flux.BatchNorm(96, Flux.relu),
	    Flux.Conv((7,7), 96=>144, bias=false), # output becomes 10x10
	    Flux.BatchNorm(144, Flux.relu),
	    Flux.Conv((7,7), 144=>192, bias=false), # output becomes 4x4
	    Flux.BatchNorm(192, Flux.relu),
	    Flux.flatten, # results in 4x4x192=3072 dimensions
	    Flux.Dense(3072, 10, bias=false),
	    Flux.BatchNorm(10)
    )
end

# ╔═╡ 3a14cb3a-7330-41e2-bd52-4f1fd0808a01
mnistSimpleCNN7()

# ╔═╡ c1697f04-5898-426b-a7ed-9f67734ceb27
function toy_model()
	return Flux.Chain(
		Flux.flatten,
		Flux.Dense(foldl(*, size(x_train[:,:,1])), 10, bias=false)
	)
end

# ╔═╡ bf0e20ca-a958-4e52-9e42-c3d70edcc270
function empiricalMSELossWithGrad(batch_size=10)
	return model -> begin
		indices = rand(1:60000, batch_size)
		return Flux.withgradient(Flux.params(model)) do
			Flux.Losses.mse(model(x_train[:,:,indices]), y_train_oh[:,indices])
		end
	end
end

# ╔═╡ df10a230-f35a-48d1-b08a-fac143b29ca4
function empiricalMSELoss(batch_size=10)
	return model -> begin
		indices = rand(1:60000, batch_size)
		return Flux.Losses.mse(model(x_train[:,:,indices]), y_train_oh[:,indices])
	end
end

# ╔═╡ 3ad0dbba-8157-4564-a68a-47d49fba43f6
struct EvalPoint
	model
	loss
end

# ╔═╡ da8513dc-ccf7-422b-92f0-6e90dc7e124d
md"# Pre-estimate parameters"

# ╔═╡ 6f4903d1-c003-4faf-b682-b582cd45f6eb
"""
Estimate Mean and Variance of empirical Loss.

The Variance of the empirical loss is given by

empLossVar = theoLossVar + sampleVar/batchSize
"""
function mean_and_variance(model_factory, samples, loss=empiricalMSELoss)
	batchSizes = [(1:10)...,100]
	losses = Array{Float64,2}(undef, (samples, length(batchSizes)))
	grad_norm = Array{Float64,2}(undef, (samples, length(batchSizes)))
	for (b_idx, batchSize) in enumerate(batchSizes)
		l_fun = loss(batchSize)
		for idx in 1:samples
			losses[idx, b_idx] = l_fun(model_factory())
			#  grad_norm[idx, b_idx] = LinearAlgebra.norm(grad)
		end
	end
	estimatedMean = Statistics.mean(vec(losses))
	vars = map(llist -> Statistics.varm(llist, estimatedMean), eachcol(losses))
	linreg = linregress(batchSizes.^(-1), vars)

	grad_norm_means = map(Statistics.mean, eachcol(grad_norm))

	# sanity check plot to check linear regression is sensible
	plt = plot(
		batchSizes.^(-1), vars,
		seriestype=:scatter,
		label="Estimated Variance of Empirical Loss of varying batch sizes",
		xlabel="1/batchSize",
		title="empirLossVar = theoLossVar + sampleVar/batchSize"
	)
	plot!(plt, [0,1], [linreg([0]), linreg([1])], label="Linear Regression")
	# ----------


	plt_norms = plot(
		batchSizes.^(-1), grad_norm_means,
		seriestype=:scatter,
		label="Estimated Variance of Empirical Loss of varying batch sizes",
		xlabel="1/batchSize",
		title="empirLossVar = theoLossVar + sampleVar/batchSize"
	)
	return (
		mean=estimatedMean, 
		theoLossVar=LinearRegression.bias(linreg),
		sampleVar=LinearRegression.slope(linreg)[1],
		plot=plt # sanity check plot
	)
end

# ╔═╡ 83d373da-561f-41b9-b5ee-2f585c6b52a9
preestimatedParams = mean_and_variance(mnistSimpleCNN7, 100)

# ╔═╡ 99c35161-c079-47d0-90d1-7c1ff0805824
val, grad = empiricalMSELossWithGrad(1000)(mnistSimpleCNN7())

# ╔═╡ 1cacb6e3-33f0-45c7-93e4-a55ad5bc4915
LinearAlgebra.norm(grad)

# ╔═╡ d1ce6753-adb8-4e15-8b00-991e7545da24
N = sum(length, Flux.params(mnistSimpleCNN7()))

# ╔═╡ e6f8b7a7-259c-45e9-8897-3be29af969c0
σ2 = preestimatedParams.theoLossVar * N

# ╔═╡ 18ba55bf-73fe-4945-9e35-c025e2af5f73
L"""
C(x,y) = \frac1N g\Bigl(-\frac{\|x-y\|^2}{2}\Bigr)
= \frac{\sigma^2}{N}\exp\Bigl(-\frac{\|x-y\|^2}{2s^2}\Bigr)
"""

# ╔═╡ 467ac6fe-a963-4a4f-90bc-966636c9997e
L"
\|\nabla \mathcal{L}(x_0)\|^2 
= \sum_{i=1}^N (\partial_i \mathcal{L}(x_0))^2 
= -g'(0) \underbrace{\frac1N\sum_{i=1}^N Y_i^2}_{\approx 1}
\approx \frac{\sigma^2}{s^2}
"

# ╔═╡ d8dc580f-d83b-490b-92ef-d2a3505356e8
L" s \approx \frac{\sigma}{\|\nabla\mathcal{L}(x_0)\|}"

# ╔═╡ b2bda984-b523-4a76-ba78-9eb7a8674581
md"# Sampling points"

# ╔═╡ 7bfc408d-b188-46c5-b52c-e65c757f9045
begin
	local dt = -pi:0.01:pi
	local plt = plot(
		sin.(dt), cos.(dt),
		xlim=(-1.3, 2),
		ylim=(-0.3, 2),
		aspect_ratio=1, 
		label=missing,
		axis=false
	)
	local ref_pt = [1/sqrt(5), 2/sqrt(5)]

	tangent(t) = ref_pt .+ t*[ref_pt[2], -ref_pt[1]]
	tpts = reshape(vcat(tangent.(dt)...), 2,:)
	plot!(
		plt, tpts[1,:], tpts[2,:], color=3, label=L"x^\perp"
	)
	
	random_pts = rand(Distributions.Normal(0, 1), 10)
	rand_directions = hcat(
		map(tangent.(random_pts)) do x
			return [x; (x/norm(x) .- x)]
		end...
	)
	plot!(
		plt, rand_directions[1,:], rand_directions[2,:], 
		seriestype=:scatter,
		color=3, label="random directions"
	)
	plot!(
		plt, rand_directions[1,:], rand_directions[2,:], 
		quiver=(rand_directions[3,:], rand_directions[4,:]),
		seriestype=:quiver,
		color=3, label="random directions"
	)

	plot!(
		plt, [0], [0],
		quiver= ([ref_pt[1]], [ref_pt[2]]),
		seriestype=:quiver, label="reference point x", color=2
	)
	plot!(
		plt, [ref_pt[1]], [ref_pt[2]],
		seriestype=:scatter, label="reference point x", color=2
	)
end

# ╔═╡ 4c49f380-d8fd-474e-b628-e53b88fd79d5
md"### Visualization of point sampling
In high dimension all random vectors are roughly orthogonal and of the same length and distance. So we can not simply sample arbitrary points. To ensure they are of varying distances their entries need to be of different variance. But this would also put them at different distances from the origin. So instead we select a reference point on the typical sphere (using the default sampling method given by simply constructing the model). Then we sample more model, but treat them as vectors originating from the reference point. We modify their variances to ensure they are of varying distance. Then we project them back to the typical distance from the origin.

As they are all orthogonal to the reference point, we expect them to land roughly in the half sphere in the reference point direction.
"

# ╔═╡ 0c69d3d9-7126-4df1-8a13-4ca57ba97b96
num_sample_points = 300

# ╔═╡ 4cc89f12-fe41-4b36-b513-5caac4a253f3
md"### Sample varying Variances"

# ╔═╡ 24c2a2d9-5591-4a4d-a2cd-d2b6f2cc570f
shape = 1

# ╔═╡ 66609fb7-5fe9-4c79-a2cb-366719402863
variances = rand(Distributions.Gamma(1/4, shape), num_sample_points)

# ╔═╡ d6069b06-cca3-45ba-9843-7f487e564acd
plot(variances, seriestype=:hist, label="variances")

# ╔═╡ 4b171e81-b4d5-4c0c-9b60-bad5095c3707
md"### Sampling points"

# ╔═╡ 93e16306-2a04-4476-8e6b-9fb42cffb29f
"""
The reference point is the `model_origin`,
sample vectors from this reference point with variances sampled above and project them onto the sphere.

Evaluate the resulting points using the `loss` passed to this function

return Array{EvalPoint} (i.e. model and loss struct)
"""
function sample_points(
	model_factory, loss=empiricalMSELoss(10); model_origin=model_factory()
)
	o_vec, restructure = Flux.destructure(model_origin)
	radius = norm(o_vec)
	print(radius) # radius of the shpere origin is on
	evalPoints = Array{EvalPoint,1}(undef, length(variances))
	@progress for (idx, var) in enumerate(variances)
		m_vec, _ = Flux.destructure(model_factory()) # pick a random direction
		m_param = o_vec .+ sqrt(var) * m_vec # move in the direction of m_vec
		m_param *= radius/norm(m_param) # put result back on the sphere

		model = restructure(m_param)
		evalPoints[idx] =  EvalPoint(model, loss(model))
	end
	return evalPoints
end

# ╔═╡ a87ac95b-c3fd-4f4f-a23a-6f8379505add
reference_pt= mnistSimpleCNN7()

# ╔═╡ d1ed8bed-6157-46cb-8616-78f30d11bfba
batchSize = 1000

# ╔═╡ 84bb3d8d-fcc6-4c52-9c25-dad5e14acf25
# evals = sample_points(
# 	mnistSimpleCNN7, 
# 	empiricalMSELoss(batchSize), 
# 	model_origin=reference_pt
# )

# ╔═╡ ef3defdc-f716-4f10-b249-e2cc7c1cdb08
distances_from_ref = map(
	x-> norm(Flux.destructure(x.model)[1] - Flux.destructure(reference_pt)[1]),
	evals
)

# ╔═╡ 966d8794-1165-4eb5-853c-2302a4001095
plot(
	distances_from_ref, seriestype=:hist, label="Distances from Reference Point",
size=(700,250))

# ╔═╡ 4c8c9b24-25c8-4b17-a648-a5edf533b7af
@bind window_size Slider(0.5:0.5:10, show_value=true, default=2)

# ╔═╡ 1de37476-c61c-419b-99c5-d838932271e1
""" return 
`dx, moving_average, moving_std`, where dx are the points around which the moving average and standarddeviations are taken. Those points are the discontinuities, in-between the average and std are constant.

Example Usage:
```julia 
dx, mv_avg, mv_std = moving_statistics(loc, val)
plot(dx, mv_avg, ribbon=mv_std)
```
"""
function moving_statistics(location, value, window_size=window_size)

	
	incr_ids = sort(1:length(location), by=idx->location[idx])
	sorted_loc = [location[idx] for idx in incr_ids]
	
	function window_around(x)
		idx_l = searchsortedfirst(sorted_loc, x-window_size)
		idx_r = searchsortedlast(sorted_loc, x+window_size)
		return [value[idx] for idx in incr_ids[idx_l:idx_r]]
	end

	# window value only changes on 
	# entry at loc-window_size` and exit at `loc+window_size`)
	# so the discontinuity points are given by
	dx = sort!(vcat(sorted_loc .- window_size, sorted_loc .+ window_size))
	dx = dx[
		# restrict range to range of locations
		searchsortedfirst(dx, sorted_loc[1]):searchsortedlast(dx, sorted_loc[end])
	]
	
	return (
		dx, 
		(x->Statistics.mean(window_around(x))).(dx),
		(x->Statistics.std(window_around(x))).(dx)
	)
end

# ╔═╡ 21493e2f-f245-41a0-9e37-ecc2ad8f2233
begin
	local plt = plot(
		distances_from_ref,
		map(x->x.loss, evals),
		seriestype=:scatter,
		title="Stationarity analysis for MNIST",
		ylabel="Loss",
		xlabel="Distance from Reference Point",
		label="Evaluation point",
		fontfamily="Computer Modern"
	)
	local dx, moving_average, moving_std = moving_statistics(
		distances_from_ref, 
		[x.loss for x in evals]
	)
	plot!(
		plt, dx, 
		moving_average,
		ribbon= moving_std,
		label = "Moving average and standard deviation (window size = $(window_size))"
	)
end

# ╔═╡ d294332e-b9f3-4eae-a042-ef632225bbec
md"# Maximum Likelihood

For the maximum likelihood we need to calculate all distances between points. Not just the distances to the reference point.
"

# ╔═╡ 544b8c66-1e98-428a-b7d7-588760949170
function pairwiseDistances(evalPts, valueMap, dist_fun=LinearAlgebra.norm)
	n = length(evalPts)
	distances = Array{Float64, 2}(undef, n, n)
	@withprogress name="Distances" begin
	for (idx, e1) in enumerate(evalPts)
		distances[1:idx,idx] = map(evals[1:idx]) do e2
			diff = valueMap(e1) - valueMap(e2)
			return dist_fun(diff)
		end
		@logprogress idx*(idx+1)/(n*(n+1))
	end
	end
	LinearAlgebra.symmetric(distances, :U)
end

# ╔═╡ ecbfe439-cedc-4cdf-8f1f-fd57591fbbb4
npoints = 100

# ╔═╡ 757d87f0-0a6c-451e-a96d-eebf31f5672e
""" Extracts the upper triangular matrix (without diagonal) as an array"""
extract_upper_tri(A) = A[LinearAlgebra.triu!(trues(size(A)), 1 )]

# ╔═╡ 62dfb376-2a0c-4757-8f6a-90210fe26753
begin 
	distances = pairwiseDistances(evals[1:npoints], e-> Flux.destructure(e.model)[1])
	sq_loss_distances = pairwiseDistances(
		evals[1:npoints], 
		e-> e.loss, 
		d->LinearAlgebra.dot(d,d)
	)
	sq_distances = pairwiseDistances(
		evals[1:npoints], 
		e-> Flux.destructure(e.model)[1],
		d->LinearAlgebra.dot(d,d)
	)
	plot(
		plot(
			extract_upper_tri(sq_distances), 
			seriestype=:hist, 
			label="Distances in between"
		),
		plot(map(x->x.loss, evals), seriestype=:hist, label="Losses"),
		size = (700, 250)
	)
end

# ╔═╡ 3d316ca4-564c-4dd3-9703-abe73436b426
function SECovariance(var, scale, noise)
	d -> var * exp(-d^2/(2*scale)) + ((d == 0) ? noise : 0)
end

# ╔═╡ 9189f132-4135-4adb-b161-b26a9b629cae
function negLogLikelihood(params, z=map(e->e.loss, evals[1:npoints]))
	mean = params[1]
	zc = z.-mean
	var = params[2]
	scale = params[3] # squared lengthscale
	noise = params[4]
	
	n = length(z)
	# want: 
	# variance * exp.(-sq_distances./(2*lengthscale)) + (noise * LinearAlgebra.I)
	#
	# but due to numerical errors, a small variance causes the determinant to be zero
	# pull this out of the determinant

	Σ = exp.(-sq_distances./(2*scale)) + (noise/var * LinearAlgebra.I)
	if (LinearAlgebra.det(Σ) <= 0) || var <= 0
		return Inf
	end

	logDet = n*log(var) + log(LinearAlgebra.det(Σ)) # calculate log(det(var * Σ))
	
	return logDet + LinearAlgebra.dot(zc, Σ\zc)/var
end

# ╔═╡ 8fb6f12c-9ce8-476d-9feb-3d5d709f415b
res = optimize(x->negLogLikelihood([
	preestimatedParams.mean, 
	preestimatedParams.theoLossVar,
	x, 
	preestimatedParams.sampleVar/batchSize
]), 0., 1000)

# ╔═╡ ef1060ca-4080-4d83-a032-965fa37283c4
begin
	local dist = extract_upper_tri(distances)
	local sqloss = extract_upper_tri(sq_loss_distances)
	local plt = plot(
		dist, sqloss,
		seriestype=:scatter, label="squared loss differences",
		ylim=(0, 3e-6)
	)

	dx, mvg_avg, mvg_std = moving_statistics(dist, sqloss)
	plot!(plt, dx, mvg_avg, ribbon=mvg_std, label="empirical variogram")
	C = SECovariance(
		preestimatedParams.theoLossVar, 
		Optim.minimizer(res), 
		preestimatedParams.sampleVar/batchSize
	)
	plot!(plt, dx, (x-> C(0)-C(x)).(dx), label="maximum likelihood over length scale")
	plt
end

# ╔═╡ d6d62078-31d4-498a-bc9e-e909f9499f9d
prelimParams = 	[
	preestimatedParams.mean, 
	preestimatedParams.theoLossVar, 
	Optim.minimizer(res), 
	preestimatedParams.sampleVar/batchSize
]

# ╔═╡ 3357f827-fba3-4878-8c7f-e899db978f19
sqrt(Optim.minimizer(res))

# ╔═╡ 5e631c3b-2c30-4dd0-a53e-2933a4a75a93
res2 = optimize(
	negLogLikelihood,
	[-Inf, 0, 0, 0],
	[Inf, Inf, Inf, Inf],
	prelimParams,
	NelderMead()
)

# ╔═╡ af391832-d142-46fa-abe3-0ed9dd7167d5
Optim.minimizer(res2)

# ╔═╡ Cell order:
# ╟─f8bb97c3-a6be-400f-8d02-548843b44aa7
# ╟─9eb1ddaa-9028-444e-8f35-218ba57d1e99
# ╠═677320e3-d796-404b-9d6b-f2e77ee947e2
# ╟─11dcf7ce-2c57-11ee-0f5b-014ecac26614
# ╠═3a14cb3a-7330-41e2-bd52-4f1fd0808a01
# ╟─c1697f04-5898-426b-a7ed-9f67734ceb27
# ╠═bf0e20ca-a958-4e52-9e42-c3d70edcc270
# ╠═df10a230-f35a-48d1-b08a-fac143b29ca4
# ╠═3ad0dbba-8157-4564-a68a-47d49fba43f6
# ╟─da8513dc-ccf7-422b-92f0-6e90dc7e124d
# ╠═6f4903d1-c003-4faf-b682-b582cd45f6eb
# ╠═83d373da-561f-41b9-b5ee-2f585c6b52a9
# ╠═99c35161-c079-47d0-90d1-7c1ff0805824
# ╠═1cacb6e3-33f0-45c7-93e4-a55ad5bc4915
# ╠═d1ce6753-adb8-4e15-8b00-991e7545da24
# ╠═e6f8b7a7-259c-45e9-8897-3be29af969c0
# ╟─18ba55bf-73fe-4945-9e35-c025e2af5f73
# ╟─467ac6fe-a963-4a4f-90bc-966636c9997e
# ╟─d8dc580f-d83b-490b-92ef-d2a3505356e8
# ╟─b2bda984-b523-4a76-ba78-9eb7a8674581
# ╟─7bfc408d-b188-46c5-b52c-e65c757f9045
# ╟─4c49f380-d8fd-474e-b628-e53b88fd79d5
# ╟─0c69d3d9-7126-4df1-8a13-4ca57ba97b96
# ╟─4cc89f12-fe41-4b36-b513-5caac4a253f3
# ╠═24c2a2d9-5591-4a4d-a2cd-d2b6f2cc570f
# ╟─66609fb7-5fe9-4c79-a2cb-366719402863
# ╟─d6069b06-cca3-45ba-9843-7f487e564acd
# ╟─4b171e81-b4d5-4c0c-9b60-bad5095c3707
# ╟─93e16306-2a04-4476-8e6b-9fb42cffb29f
# ╟─a87ac95b-c3fd-4f4f-a23a-6f8379505add
# ╠═d1ed8bed-6157-46cb-8616-78f30d11bfba
# ╟─84bb3d8d-fcc6-4c52-9c25-dad5e14acf25
# ╟─ef3defdc-f716-4f10-b249-e2cc7c1cdb08
# ╟─966d8794-1165-4eb5-853c-2302a4001095
# ╠═4c8c9b24-25c8-4b17-a648-a5edf533b7af
# ╟─1de37476-c61c-419b-99c5-d838932271e1
# ╟─21493e2f-f245-41a0-9e37-ecc2ad8f2233
# ╟─d294332e-b9f3-4eae-a042-ef632225bbec
# ╟─544b8c66-1e98-428a-b7d7-588760949170
# ╟─62dfb376-2a0c-4757-8f6a-90210fe26753
# ╠═ecbfe439-cedc-4cdf-8f1f-fd57591fbbb4
# ╟─757d87f0-0a6c-451e-a96d-eebf31f5672e
# ╟─ef1060ca-4080-4d83-a032-965fa37283c4
# ╠═3d316ca4-564c-4dd3-9703-abe73436b426
# ╠═9189f132-4135-4adb-b161-b26a9b629cae
# ╠═d6d62078-31d4-498a-bc9e-e909f9499f9d
# ╠═8fb6f12c-9ce8-476d-9feb-3d5d709f415b
# ╠═3357f827-fba3-4878-8c7f-e899db978f19
# ╠═5e631c3b-2c30-4dd0-a53e-2933a4a75a93
# ╠═af391832-d142-46fa-abe3-0ed9dd7167d5
