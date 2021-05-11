### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ d951e119-09f7-4b4e-a155-7e1ce0459317
begin
	using DataFrames
	using MixedModels, MixedModelsSim
	using PlutoUI
	using Random
end

# ╔═╡ 288bb980-b105-11eb-1687-73eaf51c6e80
md"""
# Simulating an Experiment from Scratch using `MixedModels.jl` and `MixedModelsSim.jl`

*Phillip M. Alday, 2021-05-11*

Today, we're using Pluto notebooks. If you've used Jupyter/iPython notebooks, this will seem similar. Pluto has two really cool advantages over those:
1. Pluto is *reactive*: if I change a variable anywhere in the notebook, the changes propogate throughout the notebook. So there's no mixed up state from doing things out of order.
2. Pluto files are stored as plain-text Julia-language files, so you can read and run their source code like any other source code file.

First, some setup
"""

# ╔═╡ 89b539d4-f564-4648-bf2b-7ee9c92687c8
md"""
## Assemble the Design

We're going to do a 2 x 2 x 2 design.
For concreteness, let's think of this as a linguistic design:

- **age** `old` vs. `young`, _between subjects_
- **frequency** `high` vs. `low`, _between items_
- **context** `matched` vs. `unmatched`, _within both_.

We'll leave off specifying the number of subjects and items for a moment.
"""

# ╔═╡ e3d37b13-f428-42b2-8cb5-dda5241f6181
subj_btwn = Dict(:age => ["old", "young"])

# ╔═╡ 8a23e17f-bb56-4b44-ac65-00993c948c73
item_btwn = Dict(:frequency => ["high", "low"])

# ╔═╡ 47f0e82d-abec-421a-a0a0-03d791dddf70
both_win = Dict(:context => ["matched", "unmatched"])

# ╔═╡ 099c1425-43fb-4121-b609-fc2894e344cb
md"""
For specifying the number of items and subjects, we're going to use a bit of interactivity that the Pluto notebook allows: sliders!
"""

# ╔═╡ 0ea1d41b-d225-4082-aa6b-5bc80beda96b
@bind n_subj Slider(10:100; default=40, show_value=true)

# ╔═╡ 9c989bb9-8a7e-4746-9919-1abb20a9ddef
@bind n_item Slider(10:100; default=40, show_value=true)

# ╔═╡ 823f06a3-1ba6-4c5f-b842-d2dee7bc8d3e
begin
	design = simdat_crossed(MersenneTwister(42), n_subj, n_item;
                             subj_btwn = subj_btwn,
                             item_btwn = item_btwn,
                             both_win = both_win)
	design = pooled!(DataFrame(design))
end

# ╔═╡ 04e3ba6d-83d1-4d9b-93b8-b72ed738972d
md"""
Note that `simdat_crossed` generated a column `dv` for our dependent variable that has been pre-populated with noise from a standard normal distribution ($N(0,1)$).
Typically, we will want to scale that, but we can do that in the simulation step.
Also, this dependent variable is *pure noise*: we haven't yet added in effects.
Adding in effects also comes in the simulation step.

Having the dependent variable already present and filled with noise allows us to fit a model and use the tooling around models instead of doing all the linear algebra ourselves.

So before we get to simulating, let's fit the model to the noise, just to see how things look. We're going to use effects coding for our contrasts.
"""

# ╔═╡ 95c03f4a-6e86-4fab-b706-039834b06c4d
contrasts = Dict(:age => EffectsCoding(base="young"),
                 :frequency => EffectsCoding(base="high"),
                 :context => EffectsCoding(base="matched"))

# ╔═╡ d9209d7f-8206-497f-a5dc-68c4cda53ec3
form = @formula(dv ~ 1 + age * frequency * context +
                    (1 + frequency + context | subj) +
                    (1 + age + context | item))

# ╔═╡ f4baf339-74f9-4211-b56d-83fe481977bb
m0 = fit(MixedModel, form, design; contrasts=contrasts)

# ╔═╡ 51db88e8-0658-4d5f-b299-ea1b9f1939f8
md"""
## Assemble the Random Effects

The hard part in simulating right now is specifying the random effects.
We're working on making this bit easier, but you need to specify the variance-covariance matrix of the random effects. You can see what this
looks like:
"""

# ╔═╡ 69a18181-f46e-4c9e-a79f-80db0d9fc902
vc = VarCorr(m0)

# ╔═╡ 3e3dc064-5854-4ec5-b710-a4cca1c17060
md"""
For each grouping variable (subjects and items), there are two major components: the standard deviations ahd the correlations.

Let's assume that the variability
- between items
  - in the intercept is 1.3 times the residual variability
  - in age is 0.35 times the residual variability
  - in context is 0.75 times the residual variability
- between subjects
  - in the intercept is 1.5 times the residual variability
  - in frequency is 0.5 times the residual variability
  - in context is 0.75 times the residual variability

Note these are always specified relative to the residual standard deviation.
In other words, we think about how big the between-subject and between-item differences are relative to the between-observation differences.

We can now create the associated covariance matrices.[^cholesky]

[^cholesky]: Technically, we're creating the lower Cholesky factor of these matrices, which is a bit like the matrix square root. In other words, we're creating the matrix form of standard deviations instead of the matrix form of the variances.]
"""

# ╔═╡ dc548a77-e0a5-4cfb-9371-06e48bd25f13
corr_item = [+1.0 +0.2 +0.5
	         +0.2 +1.0 -0.3
	         +0.5 -0.3 +1.0]

# ╔═╡ 872d9fcf-6ed4-4533-92b2-669a158a460c
re_item = create_re(1.3, 0.35, 0.75; corrmat=corr_item)

# ╔═╡ 90dd7f51-822d-4ebb-906c-afd9649f3e83
corr_subj = [+1.0 -0.5 +0.0
	         -0.5 +1.0 +0.8
	         +0.0 +0.8 +1.0]

# ╔═╡ b370aa4e-1341-45c5-9ab8-e4f29ef28e98
re_subj = create_re(1.5, 0.5, 0.75; corrmat=corr_subj)

# ╔═╡ 0414d540-7f4c-4231-8e01-3c097d13b538
md"""
We can check that we got these right by installing these parameter values into the model.
Note that we have to specify them in the same order as in the output from `VarCorr`.
"""

# ╔═╡ 4496e1bc-f4d7-41ac-bd36-8a75c4725fdf
begin
	# we make a copy to avoid changing the results above with reactivity
	m1 = deepcopy(m0)
	update!(m1, re_subj, re_item)
	VarCorr(m1)
end

# ╔═╡ 631ab549-c771-4974-a3d0-2c6a26e90f73
md"""
Looks good. The values don't exactly match the values in our parameter vector because the
residual standard deviation isn't exactly 1.0.

For the actual simulation, we'll need the compact form of these covariance matrices that MixedModels.jl stores uses internally.
This compact form is the parameter vector θ and we can get it back out of the model where we just installed it:
"""

# ╔═╡ f0fefbcb-51b8-458e-9fe0-96c1820e0b56
θ = m1.θ

# ╔═╡ 733ea084-7bf0-4658-9b37-dd9a749dee53
md"""
## Assemble the Fixed Effects

The last two components we need are the residual variance and the effect sizes for the fixed effects.
"""

# ╔═╡ c4666325-da0d-412e-8f2e-ae71c7f46bda
σ = 1

# ╔═╡ bc087a8a-a183-4b17-a566-4561eac5d668
β = [1.0, -1.0, 2.0, -1.5, 0.3, -1.3, 1.4, 0]

# ╔═╡ 25762800-7fea-4f2c-b6c2-978d85db45bd
md"""
The entries in the β correspond to the coefficients in the model given by
"""

# ╔═╡ 16ecfacb-f118-4358-b9b1-385f39bbc293
coefnames(m1)

# ╔═╡ 1e0c3594-dec8-4bc6-8801-70382f3df7a5
md"""
## Simulate a Single Dataset

Now we're ready to actually simulate our data.
Let's start small and simulate a single dataset and see if we're able to recover our parameter values.
"""

# ╔═╡ aa4ddaa2-f9b3-40b1-8f41-3d7478476948
begin
	# making a deepcopy here so that our previews above aren't impacted
	m_test = simulate!(MersenneTwister(42), deepcopy(m1); β=β, σ=σ)
	refit!(m_test)
end

# ╔═╡ fa18b2f3-ef07-4920-8684-8b963714545c
coef(m_test)

# ╔═╡ d0c1a83e-5b22-4edd-beb6-dcfb435d3eb5
md"""
The estimates aren't perfect, but most of them are within a single standard error of the true values, which is a good sign.
"""

# ╔═╡ 8af80b54-eff9-4a72-a269-811f7940f057
abs.(coef(m_test) - β) .< stderror(m_test)

# ╔═╡ 5f58663b-ca74-405d-b6e7-eb1a4cf856ba
md"""
## Simulate a Lot of Datasets

We can use `parametricbootstrap` to do this: the parametric bootstrap actually works by simulating new data from an existing model and then looking at how the estimates fit to that new data look.
In MixedModels.jl, you can specify different parameter values, such as the ones
 we made up for our fake data.
"""

# ╔═╡ a2e16602-7d63-4fce-aed0-55a40febd3e7
@bind n_sim Slider(10:1000; default=100, show_value=true)

# ╔═╡ ea3448a9-cb0c-4a8e-8ee0-ccf883f27511
sim = parametricbootstrap(MersenneTwister(12321), n_sim, m0; β=β, σ=σ, θ=θ)

# ╔═╡ b87b7850-8660-4730-a557-2e64d73d36d0
DataFrame(shortestcovint(sim))

# ╔═╡ c4fadf55-c23d-435b-b6b5-40a7301674a6
md"""
## See your power and profit!

Finally, we can turn this into a power table:
"""

# ╔═╡ 6cfe68cd-7183-4aa6-8011-ef5dd06ff0f7
DataFrame(power_table(sim))

# ╔═╡ 079f3bb8-511d-4993-8189-80df04ef3fc2
md"""
These are point estimates for the power. Currently, no confidence intervals or error bars are provided because there isn't a really clear ideal way to do this. (And it's not clear that "confidence interval" is really a appropriate here.)

For today's purposes, we can use the [arcsine transformation](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Arcsine_transformation) to get approximate intervals for power. For values that are close to 0 or 1, we'll use the [rule of 3](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Rule_of_three_-_for_when_no_successes_are_observed).
"""

# ╔═╡ 725425d4-dbfa-4cf6-af71-8100170cd72c
"""
	powinterval(p, n; atol=1/n)

Compute binomial 95% confidence intervals.

`p` is the estimated probability of success, `n` is the number of trials.

For `p` within `atol` of 0 or 1, the rule of three is used.
For all other `p`, the arcsine approximation is used.
"""
function powinterval(p, n; atol=1/n)

	if (1-p) < atol
		lower = 1-3/n
		upper = 1.0
	elseif p < atol
		lower = 0.0
		upper = 3/n
	else
		asinp = asin(sqrt(p))
		zterm = 1.96 / (2 * sqrt(n))

		lower = sin(asinp - zterm)^2
		upper = sin(asinp + zterm)^2
	end

	return (lower=round(lower; sigdigits=2), upper=round(upper; sigdigits=2))
end

# ╔═╡ f736b605-e460-409d-87f0-3847ac39d174
"""
	powinterval(n)

Create a function that takes a probability of success and computes a binomial confidence interval for `n` trials.
"""
powinterval(n) = p -> powinterval(p, n)

# ╔═╡ 89446f45-5cbf-4644-b9ff-a0158f4b696a
transform!(DataFrame(power_table(sim)),
		   :power => ByRow(powinterval(n_sim)) => AsTable)

# ╔═╡ 0e5bbd3e-7373-4ed1-ad3e-354f3453983a
md"""
We should also checkout how many fits were singular.
Singular fits are a sign that we actually don't have enough power to reliably distinguish the participant/item variation from the residual variation.
If we're not doing a study on individual differences, that's probably okay, but it also suggests we could simplify our model design and make our lives easier.
"""

# ╔═╡ a0fd546d-4457-4d33-9677-d724254ffdf4
count(issingular(sim))

# ╔═╡ Cell order:
# ╟─288bb980-b105-11eb-1687-73eaf51c6e80
# ╠═d951e119-09f7-4b4e-a155-7e1ce0459317
# ╟─89b539d4-f564-4648-bf2b-7ee9c92687c8
# ╠═e3d37b13-f428-42b2-8cb5-dda5241f6181
# ╠═8a23e17f-bb56-4b44-ac65-00993c948c73
# ╠═47f0e82d-abec-421a-a0a0-03d791dddf70
# ╟─099c1425-43fb-4121-b609-fc2894e344cb
# ╠═0ea1d41b-d225-4082-aa6b-5bc80beda96b
# ╠═9c989bb9-8a7e-4746-9919-1abb20a9ddef
# ╠═823f06a3-1ba6-4c5f-b842-d2dee7bc8d3e
# ╟─04e3ba6d-83d1-4d9b-93b8-b72ed738972d
# ╠═95c03f4a-6e86-4fab-b706-039834b06c4d
# ╠═d9209d7f-8206-497f-a5dc-68c4cda53ec3
# ╠═f4baf339-74f9-4211-b56d-83fe481977bb
# ╟─51db88e8-0658-4d5f-b299-ea1b9f1939f8
# ╠═69a18181-f46e-4c9e-a79f-80db0d9fc902
# ╟─3e3dc064-5854-4ec5-b710-a4cca1c17060
# ╠═dc548a77-e0a5-4cfb-9371-06e48bd25f13
# ╠═872d9fcf-6ed4-4533-92b2-669a158a460c
# ╠═90dd7f51-822d-4ebb-906c-afd9649f3e83
# ╠═b370aa4e-1341-45c5-9ab8-e4f29ef28e98
# ╟─0414d540-7f4c-4231-8e01-3c097d13b538
# ╠═4496e1bc-f4d7-41ac-bd36-8a75c4725fdf
# ╟─631ab549-c771-4974-a3d0-2c6a26e90f73
# ╠═f0fefbcb-51b8-458e-9fe0-96c1820e0b56
# ╟─733ea084-7bf0-4658-9b37-dd9a749dee53
# ╠═c4666325-da0d-412e-8f2e-ae71c7f46bda
# ╠═bc087a8a-a183-4b17-a566-4561eac5d668
# ╟─25762800-7fea-4f2c-b6c2-978d85db45bd
# ╠═16ecfacb-f118-4358-b9b1-385f39bbc293
# ╟─1e0c3594-dec8-4bc6-8801-70382f3df7a5
# ╠═aa4ddaa2-f9b3-40b1-8f41-3d7478476948
# ╠═fa18b2f3-ef07-4920-8684-8b963714545c
# ╟─d0c1a83e-5b22-4edd-beb6-dcfb435d3eb5
# ╠═8af80b54-eff9-4a72-a269-811f7940f057
# ╟─5f58663b-ca74-405d-b6e7-eb1a4cf856ba
# ╠═a2e16602-7d63-4fce-aed0-55a40febd3e7
# ╠═ea3448a9-cb0c-4a8e-8ee0-ccf883f27511
# ╠═b87b7850-8660-4730-a557-2e64d73d36d0
# ╟─c4fadf55-c23d-435b-b6b5-40a7301674a6
# ╠═6cfe68cd-7183-4aa6-8011-ef5dd06ff0f7
# ╟─079f3bb8-511d-4993-8189-80df04ef3fc2
# ╠═725425d4-dbfa-4cf6-af71-8100170cd72c
# ╠═f736b605-e460-409d-87f0-3847ac39d174
# ╠═89446f45-5cbf-4644-b9ff-a0158f4b696a
# ╟─0e5bbd3e-7373-4ed1-ad3e-354f3453983a
# ╠═a0fd546d-4457-4d33-9677-d724254ffdf4
