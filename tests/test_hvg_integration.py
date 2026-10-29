"""Synthetic-data tests; never read paper datasets or write outside temporary folders."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from scbgm.data.hvg import main

AVAILABLE = all(importlib.util.find_spec(n) for n in ('anndata','scanpy','skmisc'))

@unittest.skipUnless(AVAILABLE, 'Install data extras to run AnnData/Scanpy integration tests')
class HvgTests(unittest.TestCase):
    def setUp(self):
        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy import sparse
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        rng = np.random.default_rng(71)
        rates = rng.gamma(2, 2, (240, 450))
        x = sparse.csr_matrix(rng.poisson(rates).astype(np.int32))
        self.data = ad.AnnData(x, obs=pd.DataFrame(index=[f'cell{i}' for i in range(x.shape[0])]),
                              var=pd.DataFrame(index=[f'gene{i}' for i in range(x.shape[1])]))
        self.data.layers['counts'] = x.copy()
        self.source = self.folder/'train.h5ad'
        self.data.write_h5ad(self.source)
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def invoke(self, output='hvg.h5ad', extra=()):
        destination = self.folder/output
        with contextlib.redirect_stdout(io.StringIO()):
            main(['--dataset','synthetic','--train',str(self.source), '--output', str(destination),
                  '--n-hvg','200',*extra])
        return destination

    def test_sparse_counts_and_source_unchanged(self):
        import anndata as ad
        import numpy as np
        output = self.invoke()
        selected = ad.read_h5ad(output)
        self.assertEqual(selected.shape, (240, 200))
        self.assertEqual(selected.X.dtype, np.float32)
        self.assertFalse(selected.layers)
        self.assertTrue(selected.obs_names.equals(self.data.obs_names))
        manifest = json.loads(output.with_suffix('.json').read_text())
        self.assertEqual(manifest['source_sha256'],self.source_hash)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(),self.source_hash)

    def test_dense_and_preserved_layers(self):
        import anndata as ad
        import numpy as np
        self.data.X = self.data.X.toarray()
        self.data.write_h5ad(self.source)
        output = self.invoke(extra=['--layout','preserve','--layer','counts'])
        selected = ad.read_h5ad(output)
        self.assertIn('counts',selected.layers)
        self.assertEqual(selected.shape,(240,200))
        np.testing.assert_array_equal(selected.X.toarray(), selected.layers['counts'].toarray())
        # Exercise the dense X path separately.
        output = self.invoke('dense.h5ad')
        self.assertIsInstance(ad.read_h5ad(output).X, np.ndarray)

    def test_refuses_overwrite_even_with_wrong_sidecar(self):
        self.invoke()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            self.invoke()
        self.assertEqual(exc.exception.code,2)
        sidecar = self.folder/'blocked.json'
        sidecar.write_text('{}')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke('blocked.h5ad')
        self.assertFalse((self.folder/'blocked.h5ad').exists())

    def test_noninteger_counts_rejected_before_writing(self):
        self.data.X = self.data.X.astype(float)
        self.data.X.data[0] = 0.25
        self.data.write_h5ad(self.source)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            self.invoke()
        self.assertEqual(exc.exception.code,2)
        self.assertFalse((self.folder/'hvg.h5ad').exists())

if __name__ == '__main__':
    unittest.main()
