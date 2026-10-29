"""Dataset-independent, TRAIN-only Seurat-v3 HVG preparation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Dataset name for provenance')
    parser.add_argument('--train', type=Path, required=True, help='Raw-count TRAIN H5AD only')
    parser.add_argument('--output', type=Path, required=True, help='New output H5AD; never overwrites')
    parser.add_argument('--n-hvg', type=int, default=3000)
    parser.add_argument('--layer', help='Counts layer (default: X)')
    parser.add_argument('--layout', choices=('counts-only', 'preserve'), default='counts-only',
                        help='counts-only: clean float32 X; preserve: retain sliced layers/metadata')
    args = parser.parse_args(argv)
    if args.n_hvg <= 0:
        parser.error('--n-hvg must be positive')
    source = args.train.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f'TRAIN file does not exist: {source}')
    if output.suffix != '.h5ad':
        parser.error('--output must end in .h5ad')
    manifest_path = output.with_suffix('.json')
    if source == output or output.exists() or manifest_path.exists():
        parser.error('Output or sidecar already exists; choose a new output path')
    for key, suffix in [('MPLCONFIGDIR', 'matplotlib'), ('NUMBA_CACHE_DIR', 'numba')]:
        os.environ.setdefault(key, str(Path(tempfile.gettempdir()) / ('scbgm-' + suffix)))
    try:
        import anndata as ad
        import numpy as np
        import scanpy as sc
        from scipy import sparse
    except ImportError as exc:
        parser.error(f'Missing data dependency: {exc}. Install with: pip install -e ".[data]"')
    data = ad.read_h5ad(source)
    if args.layer:
        if args.layer not in data.layers:
            parser.error(f'Missing layer: {args.layer}')
        data.X = data.layers[args.layer].copy()
    if data.n_obs < 2 or args.n_hvg > data.n_vars:
        parser.error('Need at least two TRAIN cells and n-hvg <= number of input genes')
    values = data.X.data if sparse.issparse(data.X) else np.asarray(data.X).ravel()
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values, np.rint(values), rtol=0, atol=1e-6)):
        parser.error('HVG selection requires finite, non-negative raw integer counts')
    data.var_names_make_unique()
    sc.pp.highly_variable_genes(data, flavor='seurat_v3', n_top_genes=args.n_hvg,
                               span=0.3, inplace=True)
    genes = data.var_names[np.asarray(data.var['highly_variable'], dtype=bool)]
    if len(genes) != args.n_hvg:
        raise RuntimeError(f'Expected {args.n_hvg} HVGs, found {len(genes)}')
    if args.layout == 'preserve':
        subset = data[:, genes].copy()
    else:
        x = data[:, genes].X
        original_values = x.data if sparse.issparse(x) else np.asarray(x).ravel()
        if not np.array_equal(original_values, original_values.astype(np.float32).astype(original_values.dtype)):
            parser.error('Counts lose precision in float32; use --layout preserve')
        x = x.astype(np.float32).tocsr() if sparse.issparse(x) else np.asarray(x, dtype=np.float32)
        subset = ad.AnnData(X=x, obs=data.obs.copy(), var=data.var.loc[genes].copy())
    subset.var['hvg_rank'] = np.asarray(data.var.loc[genes, 'highly_variable_rank'], dtype=np.float64)
    selection = {'fit_split': 'TRAIN only', 'method': 'scanpy.pp.highly_variable_genes',
                 'flavor': 'seurat_v3', 'n_top_genes': args.n_hvg, 'span': 0.3}
    subset.uns['hvg_selection'] = selection
    manifest = dict(selection, dataset=args.dataset, source=str(source), output=str(output),
                    source_sha256=file_digest(source), source_shape=list(data.shape),
                    output_shape=list(subset.shape), layout=args.layout, layer=args.layer,
                    gene_order_sha256=hashlib.sha256('\n'.join(genes).encode()).hexdigest(),
                    scanpy_version=sc.__version__, final_test_loaded_during_training=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage both files first; never overwrite an existing destination, even on a concurrent run.
    with tempfile.TemporaryDirectory(prefix='.scbgm-hvg-', dir=output.parent) as staging:
        staged_h5ad = Path(staging) / 'subset.h5ad'
        staged_json = Path(staging) / 'manifest.json'
        subset.write_h5ad(staged_h5ad, compression='gzip')
        staged_json.write_text(json.dumps(manifest, indent=2) + '\n')
        os.link(staged_h5ad, output)
        try:
            os.link(staged_json, manifest_path)
        except Exception:
            output.unlink()
            raise
    print(json.dumps(manifest, indent=2))
    return 0
