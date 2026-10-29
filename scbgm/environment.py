"""Check the training environment and optionally rebuild scIB's bundled native tool."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from .utils import write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repair-scib', action='store_true',
                   help='Compile the bundled scIB C++ source in the active Python environment')
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    if not (3, 9) <= sys.version_info[:2] <= (3, 11):
        p.error('scBGM supports Python 3.9–3.11; install requirements.txt')
    import tensorflow as tf
    import tensorflow_probability as tfp
    import torch
    import scib
    import numpy as np
    pair = ('.'.join(tf.__version__.split('.')[:2]), '.'.join(tfp.__version__.split('.')[:2]))
    if pair not in {('2.15', '0.23'), ('2.10', '0.18')}:
        p.error('Use TF 2.15 / TFP 0.23 from requirements.txt, or the original TF 2.10 / TFP 0.18 environment')
    native = Path(scib.__file__).parent / 'knn_graph' / 'knn_graph.o'
    if args.repair_scib:
        compiler = shutil.which('c++') or shutil.which('g++')
        if compiler is None:
            p.error('A C++ compiler is required (macOS: xcode-select --install; Linux: install g++)')
        temporary = native.with_name('knn_graph.scbgm.tmp')
        subprocess.run([compiler, '-std=c++11', '-O3', str(native.with_suffix('.cpp')),
                        '-o', str(temporary)], check=True)
        temporary.replace(native)
    # Exercise the actual native path; a successful import alone is insufficient.
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    rng = np.random.default_rng(17)
    graph = ad.AnnData(X=np.zeros((256, 1), dtype=np.float32))
    graph.obsm['X_check'] = rng.normal(size=(256, 20)).astype('float32')
    graph.obs['batch'] = pd.Categorical(['real'] * 128 + ['generated'] * 128)
    try:
        sc.pp.neighbors(graph, use_rep='X_check', n_neighbors=10, random_state=17)
        ilisi = float(scib.me.ilisi_graph(graph, batch_key='batch', type_='knn'))
    except (OSError, FileNotFoundError) as exc:
        raise RuntimeError('scIB native tool failed. Run: scbgm setup --repair-scib') from exc
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'pandas', 'anndata', 'scanpy', 'scikit-learn',
                 'scikit-misc', 'torch', 'scib', 'keras', 'numba', 'umap-learn')}
    versions.update(tensorflow=tf.__version__, tensorflow_probability=tfp.__version__)
    result = {'python': sys.version, 'platform': platform.platform(), 'versions': versions,
              'cuda_torch': torch.cuda.is_available(),
              'tensorflow_gpus': [str(x) for x in tf.config.list_physical_devices('GPU')],
              'native_ilisi_check': ilisi}
    if args.output:
        write_json(args.output, result)
    print('Environment OK: TensorFlow, PyTorch, NB decoder and native scIB iLISI.', flush=True)
    return 0


if __name__ == '__main__':
    main()
