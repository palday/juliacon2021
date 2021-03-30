---
title: "Non-parametric Methods for Mixed-Effects Models of EEG Data"
abstract: |
   Permutation-based testing has become exceptionally popular in the analysis of neuroimaging data, but so far has depended on two-stage analyses to handle repeated measures data in order to be computationally tractable. We introduce MixedModelsPermutations to perform fast permutation-based inference on mixed models. Using mixed models instead of a two-stage approach allows us to properly represent crossed and nested designs and to escape some of the balance restrictions.
---

Permutation-based testing has become exceptionally popular in the analysis of neuroimaging data, but so far has depended on two-stage analyses to handle repeated measures data in order to be computationally tractable. 
Relying on a two-stage procedure is, however, suboptimal because it prohibits modelling more than a single random effect (and so blocks proper analysis of designs with crossed or nested random effects) and is more sensitive to issues of balance between individual groups.
With MixedModelsPermutations, we show that that fast permutation-based inference with single-stage approach based on mixed models is possible, thus opening the door for taking advantage of both permutation-based inference and mixed-effects models.
Several theoretical problems remain, such as the ideal permutation regime in a hierarchical context, but the speed of Julia allows for testing proposed solutions in a way not previously possible.
