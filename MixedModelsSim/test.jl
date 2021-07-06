### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ d951e119-09f7-4b4e-a155-7e1ce0459317
begin
	using CairoMakie # for graphics
	using DataFrames
	using Distributions
	using MixedModels, MixedModelsSim
	using Random
end

# ╔═╡ 47f0e82d-abec-421a-a0a0-03d791dddf70
conditions = Dict(:condition => [ repeat(["standard"], 4); "deviant" ] )

# ╔═╡ 0ea1d41b-d225-4082-aa6b-5bc80beda96b
n_subj = 20

# ╔═╡ 66ddef94-809c-4117-9ffa-a87790a3bc35
n_trials = 100

# ╔═╡ 823f06a3-1ba6-4c5f-b842-d2dee7bc8d3e
begin
	design = simdat_crossed(MersenneTwister(42), n_subj, n_trials;
                            item_btwn = conditions)
	design = pooled!(DataFrame(design))
end

# ╔═╡ 43215132-34fe-4fbb-8ca1-8ea9eb433402
begin
	design_bernoulli = select(design, Not(:item))
	# note that in a Bernoulli model, there is no residual variability 
 	# more technically: there is no dispersion parameter to estimate
	# so we can use whatever probability to populate the dv here
	# because it won't play a role in the simulation
	# we just need something that won't lead to total separation (i.e. no 0 nor 1)
	design_bernoulli[!, :dv] = rand(Bernoulli(0.5), nrow(design_bernoulli))
	design_bernoulli
end

# ╔═╡ 95c03f4a-6e86-4fab-b706-039834b06c4d
contrasts = Dict(:condition => EffectsCoding(base="standard"))

# ╔═╡ d9209d7f-8206-497f-a5dc-68c4cda53ec3
form = @formula(dv ~ 1 + condition + (1 + condition | subj))

# ╔═╡ f4baf339-74f9-4211-b56d-83fe481977bb
m0 = fit(MixedModel, form, design_bernoulli, Bernoulli(); contrasts=contrasts, fast=false)

# ╔═╡ 69a18181-f46e-4c9e-a79f-80db0d9fc902
VarCorr(m0)

# ╔═╡ 90dd7f51-822d-4ebb-906c-afd9649f3e83
corr_subj = [+1.0 -0.2 
			 -0.2 +1.0]

# ╔═╡ b370aa4e-1341-45c5-9ab8-e4f29ef28e98
re_subj = create_re(1.5, 1.2; corrmat=corr_subj)

# ╔═╡ 4496e1bc-f4d7-41ac-bd36-8a75c4725fdf
begin
	# we make a copy to avoid changing the results above with reactivity
	m1 = deepcopy(m0)
	# CairoMakie also has a function update!, so we have to be specific
	m1 = MixedModelsSim.update!(m1, re_subj)
	VarCorr(m1)
end

# ╔═╡ f0fefbcb-51b8-458e-9fe0-96c1820e0b56
θ = m1.θ

# ╔═╡ bc087a8a-a183-4b17-a566-4561eac5d668
# average about 75% accuracy acros conditions, lose about 10% accuracy on deviant condition
β = [1.0, -0.48] 

# ╔═╡ a2e16602-7d63-4fce-aed0-55a40febd3e7
n_sim = 100

# ╔═╡ ea3448a9-cb0c-4a8e-8ee0-ccf883f27511
sim = parametricbootstrap(MersenneTwister(12321), n_sim, m0; β=β, θ=θ)

# ╔═╡ b87b7850-8660-4730-a557-2e64d73d36d0
DataFrame(shortestcovint(sim))

# ╔═╡ 6cfe68cd-7183-4aa6-8011-ef5dd06ff0f7
DataFrame(power_table(sim))

# ╔═╡ d74266d1-48c6-4ba8-9b0b-c99b94822d02
count(issingular(sim))

# ╔═╡ 654727c7-3a88-435f-99ad-5501771923c8
sim.θ

# ╔═╡ a18886af-0fe2-49fd-a50d-5a09b4aa96f7


# ╔═╡ Cell order:
# ╠═d951e119-09f7-4b4e-a155-7e1ce0459317
# ╠═47f0e82d-abec-421a-a0a0-03d791dddf70
# ╠═0ea1d41b-d225-4082-aa6b-5bc80beda96b
# ╠═66ddef94-809c-4117-9ffa-a87790a3bc35
# ╠═823f06a3-1ba6-4c5f-b842-d2dee7bc8d3e
# ╠═43215132-34fe-4fbb-8ca1-8ea9eb433402
# ╠═95c03f4a-6e86-4fab-b706-039834b06c4d
# ╠═d9209d7f-8206-497f-a5dc-68c4cda53ec3
# ╠═f4baf339-74f9-4211-b56d-83fe481977bb
# ╠═69a18181-f46e-4c9e-a79f-80db0d9fc902
# ╠═90dd7f51-822d-4ebb-906c-afd9649f3e83
# ╠═b370aa4e-1341-45c5-9ab8-e4f29ef28e98
# ╠═4496e1bc-f4d7-41ac-bd36-8a75c4725fdf
# ╠═f0fefbcb-51b8-458e-9fe0-96c1820e0b56
# ╠═bc087a8a-a183-4b17-a566-4561eac5d668
# ╠═a2e16602-7d63-4fce-aed0-55a40febd3e7
# ╠═ea3448a9-cb0c-4a8e-8ee0-ccf883f27511
# ╠═b87b7850-8660-4730-a557-2e64d73d36d0
# ╠═6cfe68cd-7183-4aa6-8011-ef5dd06ff0f7
# ╠═d74266d1-48c6-4ba8-9b0b-c99b94822d02
# ╠═654727c7-3a88-435f-99ad-5501771923c8
# ╠═a18886af-0fe2-49fd-a50d-5a09b4aa96f7
