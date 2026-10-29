"""Public commands and their runtime import directories; archive is never imported."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
COMMANDS = {
    'generate unconditional': ('generation/unconditional/generate.py', ('training/unconditional', 'models/autoencoders/conditional')),
    'data split': ('data/prepare_shared_splits.py', ()),
    'train ae': ('training/unconditional/autoencoder.py', ()),
    'train unconditional': ('training/unconditional/train.py', ()),
    'train conditional': ('training/conditional/train_conditional.py', ('models/autoencoders/conditional',)),
    'train trajectory': ('training/trajectory/train_continuous_time_scbgm.py', ('models/autoencoders/conditional',)),
    'train counterfactual': ('training/counterfactual/train_conditional.py', ('models/autoencoders/conditional', 'models/counterfactual')),
    'generate trajectory': ('generation/trajectory/evaluate_real_d3_anchored.py', ('models/autoencoders/conditional', 'generation/trajectory', 'training/trajectory')),
    'generate conditional': ('generation/conditional/generate_conditional.py', ('models/autoencoders/conditional',)),
    'generate counterfactual': ('generation/counterfactual/generate_conditional.py', ('models/autoencoders/conditional', 'models/counterfactual')),
    'data trajectory': ('data/trajectory/prepare_wot_holdout.py', ()),
    'data counterfactual': ('data/counterfactual/prepare_arms.py', ()),
    'data rare-manifests': ('data/rare_augmentation/prepare_fraction_manifests.py', ()),
    'data rare-split': ('data/rare_augmentation/materialize_fraction_split.py', ()),
}
