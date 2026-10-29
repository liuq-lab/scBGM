"""Generate H5AD counts from a portable unconditional scBGM checkpoint."""
import argparse
import json
import tempfile
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--ae-artifact-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n-cells',type=int,default=2000)
    p.add_argument('--seed',type=int,default=3)
    p.add_argument('--decode-device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--library-mode',choices=['conditional-copula','global-empirical'],default='conditional-copula')
    args=p.parse_args(argv)
    if args.n_cells<2:p.error('n-cells must be >=2')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        p.error('Output exists; choose a new file')
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    import train as base  # dispatcher supplies the paired unconditional helper directory
    from scbgm.training.unconditional.autoencoder import load_ae_bundle, decode_sample_counts, normalize_total_log1p_dense
    from scbgm.generation.conditional.generate_conditional import sample_conditional_copula_library_sizes
    from scbgm.utils import digest,write_json
    config=json.loads((args.model_dir/'config.json').read_text())
    model_args=argparse.Namespace(**config)
    genes=np.load(args.model_dir/'genes.npy',allow_pickle=False).astype(str)
    # Original AE stores a pandas-derived object array. These are local training artifacts.
    ae_genes=np.load(args.ae_artifact_dir/'genes.npy',allow_pickle=True).astype(str)
    if not np.array_equal(genes,ae_genes):raise ValueError('AE/model gene order differs')
    bundle=load_ae_bundle(args.ae_artifact_dir,device=args.decode_device)
    base.configure_tensorflow();base.seed_everything(args.seed)
    base.install_generator_variance_epsilon(model_args.generator_variance_eps)
    with tempfile.TemporaryDirectory() as folder:
        model=base.build_bgm(model_args,Path(folder))
        model.g_net.load_weights(str(args.model_dir/'generator.weights.h5'))
        z=np.random.default_rng(args.seed).standard_normal((args.n_cells,model_args.bgm_z_dim)).astype(np.float32)
        standardized,_,_=base.generate_bgm_features(model,z,model_args,args.seed+1)
    with np.load(args.model_dir/'latent_scale.npz') as scale:
        latent=standardized*scale['sd'][None,:]+scale['mean'][None,:]
    if latent.shape[1]!=bundle.model.latent_dim:raise ValueError('Generator/AE latent dimensions differ')
    if args.library_mode=='conditional-copula':
        libraries,_,policy=sample_conditional_copula_library_sizes(bundle.train_latent,bundle.train_library_size,
            latent,args.seed+2,pca_components=32,knn_k=64,n_jobs=1)
    else:
        libraries=np.random.default_rng(args.seed+2).choice(bundle.train_library_size,args.n_cells)
        policy={'policy':'global-empirical','fit_split':'TRAIN only'}
    counts=decode_sample_counts(bundle.model,latent,libraries,batch_size=256,device=args.decode_device,seed=args.seed+3)
    generated=ad.AnnData(normalize_total_log1p_dense(counts),
        obs=pd.DataFrame(index=[f'generated_{i:06d}' for i in range(len(counts))]),var=pd.DataFrame(index=genes))
    generated.layers['counts']=sparse.csr_matrix(counts)
    generated.obsm['X_ae']=latent
    args.output.parent.mkdir(parents=True,exist_ok=True)
    generated.write_h5ad(args.output,compression='gzip')
    write_json(args.output.with_suffix('.json'),{'model_dir':str(args.model_dir.resolve()),
        'generator_sha256':digest(args.model_dir/'generator.weights.h5'),
        'ae_sha256':digest(args.ae_artifact_dir/'model.pt'),'seed':args.seed,'n_cells':args.n_cells,
        'X_space':'normalize_total(10000)+log1p','counts_layer':'counts','library_policy':policy})
if __name__=='__main__':main()
