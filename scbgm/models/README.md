# Model code

- `bayesgm/`: the vendored BayesGM library and its license.
- `autoencoders/conditional/`: AE helper used by the conditional, trajectory and counterfactual workflows.
- `counterfactual/label_factors.py`: label-factor model for disease-state counterfactuals.

The unconditional AE stays in `training/unconditional/autoencoder.py` because
`train.py` imports it from its own folder. The two AE implementations serve
different workflows and should not be merged.
