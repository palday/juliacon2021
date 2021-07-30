### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ bdcfbfee-dde9-11eb-2c56-abdd9aa31f48
begin
	using AlgebraOfGraphics
	using CairoMakie
	using DataFrames
	using MixedModels
	using MixedModelsMakie
	using MixedModelsSim
	using MixedModelsPermutations
	using StatsBase
	using Random
	
	using AlgebraOfGraphics: density
	using MixedModelsPermutations: resample!, permute!
	
	using PlutoUI; PlutoUI.TableOfContents()
end

# ╔═╡ 074276a8-90d0-4b6f-9703-5f1cc3961a5d
md"""# Simulate some data"""

# ╔═╡ 1c2b9b8a-3606-4946-898d-ae128f0101e8
begin
	n_subj = 40
	n_item = 80
	design = simdat_crossed(MersenneTwister(42), n_subj, n_item;
							subj_btwn = Dict(:age => ["old", "young"]),
							item_btwn = Dict(:frequency => ["high", "low"]),
							both_win = Dict(:context => ["matched", "unmatched"]))
	design = pooled!(DataFrame(design))
	signedsqrt(x) = sign(x) * sqrt(abs(x))
	# nonlinear transform to break normality
	transform!(design, :dv => ByRow(signedsqrt) => :err)
end

# ╔═╡ 473d917c-fdd6-4702-9147-8c48ebd86cb8
β = [1.0, -1.0, 2.0, -1.5, 0.3, -1.3, 1.4, 0]

# ╔═╡ 3e22a7b7-18ac-4c68-8e67-3c9772e6289e
model = let  
	# example adapted from MixedModelsSim
	contrasts = Dict(:age => EffectsCoding(base="young"),
					 :frequency => EffectsCoding(base="high"),
					 :context => EffectsCoding(base="matched"))
	form = @formula(dv ~ 1 + age * frequency * context +
                    (1 + frequency + context | subj) +
                    (1 + age + context | item))
	re_item = create_re(1.3, 0.35, 0.75)
	re_subj = create_re(1.5, 0.5, 0.75)
	σ = 1
	m = LinearMixedModel(form, design; contrasts=contrasts)
	MixedModelsSim.update!(m, re_subj, re_item)
	simulate!(MersenneTwister(24), m; β=β, σ=σ)
	
	# remove the normally distributed error and 
	# add in some non-normally diostributed error
	transform!(design, :err => Base.Fix1(+, fitted(m)) => :y) 
	refit!(m, design.y)
	design[!, :resid] = residuals(m)
	m
end
	

# ╔═╡ fafee64a-714f-4fe9-aac2-4aa90913a6fe
md"""# Parametric Assumptions Break Down"""

# ╔═╡ a2181264-4834-47e3-821d-56566d42ebaf
let plt = data(design)
	plt *= mapping(:resid => "Residual (observation level) error"; 
				   col=:age, row=:frequency, color=:context)
	plt *= density()
	draw(plt)
end
	

# ╔═╡ ff05361f-199b-4558-a037-c44f26871537
md"""
The residuals are clearly not normally distributed, so confidence intervals based on a parametric assumption of normality will not be accurate. This includes both the "Wald" estimates (±1.96*SE) and even the parametric bootstrap.
"""

# ╔═╡ bd7bc89d-481e-450a-ae27-f5987e714fff
pboot = parametricbootstrap(MersenneTwister(1), 250, model; use_threads=false)

# ╔═╡ 5b2d3a64-74e3-4f36-becd-94b0f75c802a
pbootci = DataFrame(shortestcovint(pboot))

# ╔═╡ a98945e4-c752-435f-a9d1-341cb8bd25b9
md"""
The reason for this failure becomes apparent if we examine the residuals from one of the parametric iterations: The simulated data don't have the same conditional distribution as the original data.
"""

# ╔═╡ fcad67b2-e4ee-4523-8a12-005266e68549
let pltdata = copy(design)
	pltdata[!, :resid] = residuals(refit!(simulate!(MersenneTwister(1), deepcopy(model))))
	plt = data(pltdata)
	plt *= mapping(:resid => "Residual (observation level) error"; 
				   col=:age, row=:frequency, color=:context)
	plt *= density()
	draw(plt)
end

# ╔═╡ 7dce0d11-e515-43d4-9e43-d1f100c05f4f
md"""
Usually at this point, we would consider using nonparametric methods, such as various resampling methods like the nonparametric bootstrap or a permutation test.
"""

# ╔═╡ 8dcb405d-4e80-4825-8874-d305a081e8c8
md"""
# Resampling Stratified Data

What does it mean to resample or permute multilevel/hierarchical/stratified data? When there is a clean nesting structure and balanced, you can just resample within each level (as is done in stratified cross-validation), but what happens if you have partially or full crossed data or imbalance or all of these things? Traditional analyses at each level in the hierarchy ignore the structure of the data.

Mixed-effects models provide a great way to model these data -- can we somehow use them to construct a resampling procedure? We indeed can. The procedure is based on treating all the variance components, including the residual error, as places where resampling methods can be applied. Following the ter Braak procedure, the observation-level, i.e., the residuals are resampled. (For a classical OLS model, this is equivalent to resampling entire observations.) Building upon this, the conditional modes / BLUPs, i.e. the random effects, are resampled.

Using this, we can compute a nonparametric bootstrap.
"""

# ╔═╡ b1371cc8-32cc-4736-b262-e193fba9fb09
npboot = nonparametricbootstrap(MersenneTwister(1), 250, model; use_threads=false)

# ╔═╡ 63b96be1-5476-441a-ba22-029f52242e88
npbootci = DataFrame(shortestcovint(npboot))

# ╔═╡ 1bd430f1-7c1a-4fc5-bb05-b5a08fd2a2b4
md"""One important property of this method is that the bootstrap repetitions maintain the structure of the original data (because they are resampled) and don't assume the parametric structure (as parametric simulation does)."""

# ╔═╡ c0f308c5-9581-465b-98c9-777f158ed6af
let pltdata = copy(design)
	pltdata[!, :resid] = residuals(refit!(resample!(MersenneTwister(1), deepcopy(model))))
	plt = data(pltdata)
	plt *= mapping(:resid => "Residual (observation level) error"; 
				   col=:age, row=:frequency, color=:context)
	plt *= density()
	draw(plt)
end

# ╔═╡ 4d450678-523b-4c0e-b233-215e4b410d04
md"""
# Even more to come!

The same logic can be extended from sampling with replacement (i.e. the bootstrap) to sampling without replacement (i.e. permutation tests), which are common in neuroscience for mass-univariate testing, including cluster-based procedures. There are a few challenges though: the shrinkage inherent in the random effects can create problems for resampling without replacement."""

# ╔═╡ 0db61432-267d-4445-8c25-9f7ae6bc3895
shrinkageplot(model, :subj)

# ╔═╡ faa96d8e-e5eb-4527-9d2e-a2a963168bde
shrinkageplot(model, :item)

# ╔═╡ e4c3eef2-c29d-4fe2-aa13-5b0c09a6bf80
md"""We have developed a number of techniques for "re-inflating" these estimates and our simulations suggest that we are able to achieve nominal error rates.

Julia has been a real boost here, both the language and its speed. We are able to prototype different alternatives quickly, then run large simulation to examine their properties, in a way that would have been onerous in other languages. Moreoever, we are able to perform permutation tests fast enough on real neuroscience data (e.g., EEG), that we feel that we can convince users that it's possible to have your cake and eat it too by combining the power of mixed-effects models with the data-driven opimization of cluster-based permutation tests."""

# ╔═╡ 299a91e8-2867-4d16-8733-d13aa1f055ea
md"""

# Acknowledgements

The development of this package was supported by the Center for Interdisciplinary Research (ZiF), Bielefeld, as part of the Cooperation Group "Statistical models for psychological and linguistic data".

This package was developed in collaboration with Jaromil Frossard, Benedikt Ehinger and Olivier Renaud.
"""

# ╔═╡ Cell order:
# ╠═bdcfbfee-dde9-11eb-2c56-abdd9aa31f48
# ╟─074276a8-90d0-4b6f-9703-5f1cc3961a5d
# ╠═1c2b9b8a-3606-4946-898d-ae128f0101e8
# ╠═473d917c-fdd6-4702-9147-8c48ebd86cb8
# ╠═3e22a7b7-18ac-4c68-8e67-3c9772e6289e
# ╟─fafee64a-714f-4fe9-aac2-4aa90913a6fe
# ╠═a2181264-4834-47e3-821d-56566d42ebaf
# ╟─ff05361f-199b-4558-a037-c44f26871537
# ╠═bd7bc89d-481e-450a-ae27-f5987e714fff
# ╠═5b2d3a64-74e3-4f36-becd-94b0f75c802a
# ╟─a98945e4-c752-435f-a9d1-341cb8bd25b9
# ╠═fcad67b2-e4ee-4523-8a12-005266e68549
# ╟─7dce0d11-e515-43d4-9e43-d1f100c05f4f
# ╟─8dcb405d-4e80-4825-8874-d305a081e8c8
# ╠═b1371cc8-32cc-4736-b262-e193fba9fb09
# ╠═63b96be1-5476-441a-ba22-029f52242e88
# ╟─1bd430f1-7c1a-4fc5-bb05-b5a08fd2a2b4
# ╠═c0f308c5-9581-465b-98c9-777f158ed6af
# ╟─4d450678-523b-4c0e-b233-215e4b410d04
# ╠═0db61432-267d-4445-8c25-9f7ae6bc3895
# ╠═faa96d8e-e5eb-4527-9d2e-a2a963168bde
# ╟─e4c3eef2-c29d-4fe2-aa13-5b0c09a6bf80
# ╟─299a91e8-2867-4d16-8733-d13aa1f055ea
