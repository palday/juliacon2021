### A Pluto.jl notebook ###
# v0.14.8

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

# ╔═╡ f9949a62-b108-11eb-37d6-bb0790f7899a
begin
	using DataFrames
	using LRUCache
	using MixedModels, MixedModelsSim
	using PlutoUI
	using Random
	cache = LRU(;maxsize=10_000)
	nothing
end

# ╔═╡ da2e5642-3188-4f0d-8a8a-402a64860366
begin 
	both_win = Dict(:context => ["matched", "unmatched"])
	item_btwn = Dict(:frequency => ["high", "low"])
md"""
Let's look at a simple 2x2 design:
- **frequency** `high` vs. `low`, _between items_
- **context** `matched` vs. `unmatched`, _within both_.
"""
end

# ╔═╡ d2584c26-9db7-4f4e-afbc-9c1626130a16
# Set the contrasts
contrasts = Dict(:frequency => EffectsCoding(base="high"),
                 :context => EffectsCoding(base="matched"))

# ╔═╡ 039bcc98-70d7-452f-9fef-785e848c6917
form = @formula(dv ~ 1 + frequency * context +
                    (1 + frequency * context | subj) +
                    (1 + context | item))

# ╔═╡ 02682ad7-c4d9-461d-b5ac-979ca2d8e293
md"Number of subjects"

# ╔═╡ 99969868-a8d9-4096-ae05-8025f146fbd7
@bind n_subj NumberField(10:100; default=40)

# ╔═╡ 0d167b82-beae-4dd2-8b7f-8e18d68f865a
md"Number of items"

# ╔═╡ 8be15f3f-1fba-4f74-aeae-1aae66ba4689
@bind n_item  NumberField(10:100; default=40)

# ╔═╡ 7db32c88-00ba-4d87-9129-71e63c4ba1d1
md"**Relative** By-subject standard deviation for `Intercept`"

# ╔═╡ 5a015198-43c5-4c9e-a5b5-ae7821df1ae2
@bind s_subj_intercept NumberField(0:0.1:5; default=1)

# ╔═╡ 8f064745-46b0-46b6-952c-8ae4cd27a408
md"**Relative** By-subject standard deviation for `context`"

# ╔═╡ 4f6a4441-53b9-4d1e-b61a-da839f85d1a2
@bind s_subj_context NumberField(0:0.1:5; default=1)

# ╔═╡ 55e9fbaa-535e-4c77-8105-4402718d5e5f
md"**Relative**  By-subject standard deviation for `frequency`"

# ╔═╡ 8ea30194-1ad1-44bb-8185-8236aa6c86eb
@bind s_subj_frequency  NumberField(0:0.1:5; default=1)

# ╔═╡ 34b7f4a2-3e81-4b3e-8c32-77894e3f88a1
md"**Relative**  By-subject standard deviation for `frequency`-`context` interaction"

# ╔═╡ fe53bb70-b1eb-4858-98fa-2256e8399b5b
@bind s_subj_interaction  NumberField(0:0.1:5; default=1)

# ╔═╡ 5344e586-3bb1-44ce-a354-f5f363e5022e
md"**Relative** By-item standard deviation for `Intercept`"

# ╔═╡ fcbf836f-5e8e-4763-9c0f-ad9f9c3b5776
@bind s_item_intercept  NumberField(0:0.1:5; default=1)

# ╔═╡ bc85fffd-d37c-4249-b119-cb2d3d625f39
md"**Relative** By-item standard deviation for `Context`"

# ╔═╡ a85ac4b0-1259-4a87-ac20-cca6a8924ba4
@bind s_item_context  NumberField(0:0.1:5; default=1)

# ╔═╡ 3287c778-ce89-42e7-b6c9-44da71590b71
β = [-2, -1, -2, -1.2]

# ╔═╡ 701566c0-e28d-4e7f-bc45-191b533c8c2c
md"Residual standard deviation"

# ╔═╡ b6a549f9-acff-40a1-98fd-41affd1fd3c3
@bind σ  NumberField(0:0.1:5; default=1)

# ╔═╡ 892ad822-4af3-4410-8e27-46ee175ac513
md"Number of simulations"

# ╔═╡ c60759a8-729d-4c1d-84bc-718abe575144
@bind n_sim  NumberField(100:100:1000; default=100)

# ╔═╡ dd586970-9cb9-425e-8492-af194730c371
get!(cache, (contrasts, n_sim, n_subj, n_item, s_item_intercept, s_item_context, s_subj_intercept, s_subj_frequency, s_subj_context, s_subj_interaction, β, σ)) do 
	global m0
	design = simdat_crossed(MersenneTwister(42), n_subj, n_item;
                             item_btwn = item_btwn,
                             both_win = both_win)
	design = pooled!(DataFrame(design))
	
	m0 = fit(MixedModel, form, design; contrasts=contrasts)
	re_item = create_re(s_item_intercept, s_item_context)
	re_subj = create_re(s_subj_intercept, s_subj_frequency, s_subj_context, s_subj_interaction)
		if string(m0.reterms[1]) == "subj"
		update!(m0, float.(re_subj), float.(re_item))
	else	
		update!(m0, float.(re_item), float.(re_subj))
	end
	sim = parametricbootstrap(MersenneTwister(42), n_sim, m0; β=β, σ=σ, θ=m0.θ, use_threads=true)
	return DataFrame(power_table(sim))
end

# ╔═╡ Cell order:
# ╟─f9949a62-b108-11eb-37d6-bb0790f7899a
# ╟─da2e5642-3188-4f0d-8a8a-402a64860366
# ╟─d2584c26-9db7-4f4e-afbc-9c1626130a16
# ╠═039bcc98-70d7-452f-9fef-785e848c6917
# ╟─02682ad7-c4d9-461d-b5ac-979ca2d8e293
# ╟─99969868-a8d9-4096-ae05-8025f146fbd7
# ╟─0d167b82-beae-4dd2-8b7f-8e18d68f865a
# ╟─8be15f3f-1fba-4f74-aeae-1aae66ba4689
# ╟─7db32c88-00ba-4d87-9129-71e63c4ba1d1
# ╟─5a015198-43c5-4c9e-a5b5-ae7821df1ae2
# ╟─8f064745-46b0-46b6-952c-8ae4cd27a408
# ╟─4f6a4441-53b9-4d1e-b61a-da839f85d1a2
# ╟─55e9fbaa-535e-4c77-8105-4402718d5e5f
# ╟─8ea30194-1ad1-44bb-8185-8236aa6c86eb
# ╟─34b7f4a2-3e81-4b3e-8c32-77894e3f88a1
# ╟─fe53bb70-b1eb-4858-98fa-2256e8399b5b
# ╟─5344e586-3bb1-44ce-a354-f5f363e5022e
# ╟─fcbf836f-5e8e-4763-9c0f-ad9f9c3b5776
# ╟─bc85fffd-d37c-4249-b119-cb2d3d625f39
# ╟─a85ac4b0-1259-4a87-ac20-cca6a8924ba4
# ╠═3287c778-ce89-42e7-b6c9-44da71590b71
# ╟─701566c0-e28d-4e7f-bc45-191b533c8c2c
# ╟─b6a549f9-acff-40a1-98fd-41affd1fd3c3
# ╟─892ad822-4af3-4410-8e27-46ee175ac513
# ╟─c60759a8-729d-4c1d-84bc-718abe575144
# ╟─dd586970-9cb9-425e-8492-af194730c371
