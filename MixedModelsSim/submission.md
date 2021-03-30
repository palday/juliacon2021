---
title: "Fast Simulation-Based Power Analyses for Mixed-Effects Models"
abstract: |
   Simulation is the gold standard for power analysis of mixed-effects models.
    Traditionally, realistic simulations have been both hard to set up and computationally intractable.
    Here we present a Julia-based workflow for fast and easy power analysis of mixed-effects models.
    With MixedModelsSim.jl, we can easily simulate realistic datasets, which we can then quickly analyze with MixedModels.jl.
    Combined with Pluto.jl, users can explore the impact on power of design decisions interactively.
---

For this demonstration, we will use MIxedModelsSim.jl to quickly simulate a dataset from scratch, including both fixed and random effects, then use that simulated data to perform a power analysis. We will show how the speed in Julia enables rapid exploration of the experimental design space -- for example whether it's better to include more items or more subjects in a repeated-measures design.  We will briefly touch upon a few advanced topics, such as
- how singular model fits can arise and what this means for power analysis (e.g., what happens when a random effect goes to zero in practice even when that component is known not to be zero in theory)
- the impact of correlation in the random effects and how that influences statistical power, as well as the impact of tools such as forcing the estimated correlations to be zero (with `zerocorr`)
- how to extend these approaches to generalized linear mixed models for e.g. logistic regression
