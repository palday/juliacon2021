### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 97b1440a-ddd4-11eb-241e-676503c17b63
begin
	using AlgebraOfGraphics
	using CairoMakie
	using DataFrames
	using Effects
	using GLM
	using RDatasets
	using StandardizedPredictors
	using Statistics
	using StatsBase
	
	using PlutoUI
	TableOfContents()
end


# ╔═╡ f1066822-8350-4311-97c2-f64f4b3ddddf
md"""
# Data

The data come from the US General Social Surveys between 1972 and 2016.

- Year: Year of the Survey
- Sex: Sex of the respondent 
- Education: Education in Years
- Vocabulary: Vocabulary test score; number correct out of 10
"""

# ╔═╡ 1e51e49d-be58-473a-b7b5-16b5674bcafa
begin
	dat = select!(RDatasets.dataset("car", "Vocab"), Not(:Timestamp))
	describe(dat)
end

# ╔═╡ 59fee477-dd15-4789-9dc8-d1db23ffba59
let pltdat = combine(groupby(dat, [:Sex, :Education]), 
					 :Vocabulary => mean, :Vocabulary => sem => :sem;
					 renamecols=false),
	plt = data(pltdat)
	plt *= mapping(:Education, :Vocabulary; color=:Sex) * 
			(visual(Scatter) + mapping(:sem) * visual(Errorbars))
	draw(plt)
end

# ╔═╡ 9b18fd74-d5c4-426c-a07a-72c127061559
md"""# Two-Way Interaction Model"""

# ╔═╡ adc39a8c-67d8-4c26-abb4-242ad420a035
m1 = lm(@formula(Vocabulary ~ Education * Sex), dat)

# ╔═╡ de5702a7-6900-438d-b078-18ddf9145d88
md"""## Getting to know Effects and Typical Values"""

# ╔═╡ 569052bd-c256-47e6-8465-c352025fae10
let 
	design = Dict(:Education => sort(unique(dat.Education)), :Sex => ["Male", "Female"])
	effects(design, m1)
end

# ╔═╡ 06ab92be-2be6-4397-960f-93990d61e202
let 
	design = Dict(:Education => sort(unique(dat.Education)))
	effects(design, m1)
end

# ╔═╡ a4b97c35-f10f-401f-8577-4882bc580521
let 
	design = Dict(:Sex => ["Male", "Female"])
	effects(design, m1)
end

# ╔═╡ 54ddf5cd-a74e-433d-9b55-77b7336efdd0
let 
	design = Dict(:Sex => ["Male", "Female"])
	effects(design, m1; typical=median)
end

# ╔═╡ 8e59155f-9a8a-4135-b438-ce3d0b286800


# ╔═╡ ed86a0a4-ad35-417b-bf6d-9c9c8a960771
md"""## Predictions and Effects are Invariant to Contrast Coding"""

# ╔═╡ 783c8fa1-b57d-4225-9ff3-1a743a756fea
m2 = lm(@formula(Vocabulary ~ 1 + Education * Sex), dat; contrasts=Dict(:Education => Center(12), :Sex => EffectsCoding()))

# ╔═╡ 6ab0d433-c25e-4a50-bbc7-79fb8c8feab0
let 
	design = Dict(:Sex => ["Female", "Male"])
	effects(design, m2)
end

# ╔═╡ 0d827b0b-f2b1-4e8d-a5e5-93ace4092df5
md"""# Three-Way Interaction Model"""

# ╔═╡ 96c53ec3-3269-4b41-ae7f-58df7331f411
m3 = lm(@formula(Vocabulary ~ 1 + Year* Education * Sex), dat; dropcollinear=false)

# ╔═╡ 9337116c-77d4-463d-bff9-e5e3d98cf4a2
let 
	design = Dict(:Education => 1:12, :Sex => ["Female", "Male"])
	effects(design, m3)
end

# ╔═╡ cc1c3dc6-da85-4ff4-83ec-2ba4108f44c0
let
	design = Dict(:Education => 1:12, :Sex => ["Female", "Male"])
	refgrid = effects(design, m3)
	plt = data(refgrid)
	plt *= mapping(:Education, :Vocabulary; lower=:lower, upper=:upper, color=:Sex)
	plt *= (visual(Lines) + visual(LinesFill))
	draw(plt)
end

# ╔═╡ fa97440e-659c-4f30-a362-d48113123af3
let years = filter(in(collect(1974:4:2000)), unique(dat.Year))
	design = Dict(:Education => 1:10, :Sex => ["Female", "Male"], 
		          :Year => years)
	refgrid = effects(design, m3)
	refgrid[!, :YearStr] = string.(refgrid.Year)
	plt = data(refgrid)
	plt *= mapping(:Education, :Vocabulary; lower=:lower, upper=:upper, color=:Sex,
				   layout=:YearStr)
	plt *= (visual(Lines) + visual(LinesFill))
	draw(plt)
end

# ╔═╡ 633f2b08-a400-4bee-8a4a-37a69cffdbf7
let years = filter(in(collect(1974:4:2000)), unique(dat.Year))
	design = Dict(:Education => unique(dat.Education), 
				  :Sex => ["Female", "Male"], 
		          :Year => years)
	refgrid = effects(design, m3)
	refgrid[!, :YearStr] = string.(refgrid.Year)
	effplt = data(refgrid)
	effplt *= mapping(:Education, :Vocabulary; lower=:lower, upper=:upper, color=:Sex,
				      layout=:YearStr)
	effplt *= (visual(Lines) + visual(LinesFill))
	
	
	aggdat = combine(groupby(dat, [:Sex, :Education, :Year]), 
					 :Vocabulary => mean, 
					 :Year => ByRow(string) => :YearStr;
					 renamecols=false)
	aggdat[!, :Sex] = string.(aggdat.Sex) # for string comparison in combining plots
	filter!(:Year => in(years), aggdat)
	datplt = data(aggdat) 
	datplt *= mapping(:Education, :Vocabulary; color=:Sex, layout=:YearStr) 
	datplt *= visual(Scatter; alpha=0.1)
	draw(datplt + effplt)
end

# ╔═╡ Cell order:
# ╠═97b1440a-ddd4-11eb-241e-676503c17b63
# ╟─f1066822-8350-4311-97c2-f64f4b3ddddf
# ╠═1e51e49d-be58-473a-b7b5-16b5674bcafa
# ╠═59fee477-dd15-4789-9dc8-d1db23ffba59
# ╟─9b18fd74-d5c4-426c-a07a-72c127061559
# ╠═adc39a8c-67d8-4c26-abb4-242ad420a035
# ╟─de5702a7-6900-438d-b078-18ddf9145d88
# ╠═569052bd-c256-47e6-8465-c352025fae10
# ╠═06ab92be-2be6-4397-960f-93990d61e202
# ╠═a4b97c35-f10f-401f-8577-4882bc580521
# ╠═54ddf5cd-a74e-433d-9b55-77b7336efdd0
# ╠═8e59155f-9a8a-4135-b438-ce3d0b286800
# ╟─ed86a0a4-ad35-417b-bf6d-9c9c8a960771
# ╠═783c8fa1-b57d-4225-9ff3-1a743a756fea
# ╠═6ab0d433-c25e-4a50-bbc7-79fb8c8feab0
# ╟─0d827b0b-f2b1-4e8d-a5e5-93ace4092df5
# ╠═96c53ec3-3269-4b41-ae7f-58df7331f411
# ╠═9337116c-77d4-463d-bff9-e5e3d98cf4a2
# ╠═cc1c3dc6-da85-4ff4-83ec-2ba4108f44c0
# ╠═fa97440e-659c-4f30-a362-d48113123af3
# ╠═633f2b08-a400-4bee-8a4a-37a69cffdbf7
