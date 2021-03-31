---
title: "Non-parametric Methods for Mixed-Effects Models of EEG Data"
abstract: |
   Permutation-based testing has become exceptionally popular in the analysis of neuroimaging data, but so far has depended on two-stage analyses to handle repeated measures data in order to be computationally tractable. We introduce MixedModelsPermutations to perform fast permutation-based inference on mixed models. Using mixed models instead of a two-stage approach allows us to properly represent crossed and nested designs, explicitly modelling multiple sources of variability and unbalanced designs.
---

Permutation-based testing has become exceptionally popular in the analysis of neuroimaging data, but so far has depended on two-stage analyses to handle repeated measures data in order to be computationally tractable. 
Relying on a two-stage procedure is, however, suboptimal because it prohibits modelling more than a single random effect (and so blocks proper analysis of designs with crossed or nested random effects) and is more sensitive to issues of balance between individual groups.
This precluded the wide adoption of MixedModels for neuroimaging, where variability between stimuli (item-effects) are very common.
With MixedModelsPermutations, we show that fast permutation-based inference with a single-stage approach based on mixed models is possible, thus opening the door for taking advantage of both permutation-based inference and mixed-effects models.
Several theoretical problems remain, such as the ideal permutation regime in a hierarchical context, but the speed of Julia allows for testing proposed solutions in a way not previously possible.
