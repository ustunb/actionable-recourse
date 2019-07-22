This folder contains scripts and notebooks showing how to use the `recourse` package.

## Examples

* `/ex_01_quickstart.ipynb`

This example shows a light walkthrough of the basic concepts in setting up Recourse. We set up an `ActionSet` object and show some custom adjustments one can make to the `ActionSet` object to alter the bounds, direction and step-size. We then train a classifier and generate a single actionset for sample individuals. We show that these actionsets flip the prediction of the individuals under the model.

* `/ex_02_audit_model-effects-of-recourse-demo.ipynb`

This example takes the previous example a step further and uses the `RecourseAuditor` class to run a recourse audit over an entire sample population. We recreate Demo. #1 in our paper.

* `/ex_03_audit_out-of-sample-demo.ipynb`

We take the previous example a step further and use the `Flipset` class to generate an entire flipset (a "flipset" is a set of `k` actionsets for an individual, in case the "least expensive" actionset according to our definition of cost is out of reach for the individual for unknown reasons). We recreate Demo. #2 in our paper.

* `/ex_04_audit_demo-disparities-in-recourse.ipynb`

We use the concepts learned in previous examples to (1) show how recourse costs differ across populations and across different training sets (2) generate flipsets for individuals in each population. In doing so, we recreate Demo. #3 in our paper.
