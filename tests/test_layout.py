import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from scbgm.registry import ROOT
from scbgm.audit import audit

class LayoutTests(unittest.TestCase):
    def test_first_party_package_import_targets_exist(self):
        for path in ROOT.rglob('*.py'):
            if 'bayesgm' in path.parts: continue
            for node in ast.walk(ast.parse(path.read_text())):
                names=[]
                if isinstance(node,ast.ImportFrom) and node.module: names=[node.module]
                elif isinstance(node,ast.Import): names=[a.name for a in node.names]
                for name in names:
                    if not name.startswith('scbgm.'): continue
                    target=ROOT.parent/Path(*name.split('.'))
                    self.assertTrue(target.is_dir() or target.with_suffix('.py').is_file(),
                                    f'{path}: unresolved {name}')

    def test_installed_runtime_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            package=Path(directory)/'scbgm'
            shutil.copytree(ROOT,package,ignore=shutil.ignore_patterns('__pycache__'))
            env=os.environ.copy();env['PYTHONPATH']=directory;env['PYTHONDONTWRITEBYTECODE']='1'
            for args in [['list'],['doctor'],['data','hvg','--help']]:
                child=subprocess.run([sys.executable,'-m','scbgm',*args],cwd=directory,env=env,
                                     capture_output=True,text=True)
                self.assertEqual(child.returncode,0,child.stdout+child.stderr)
            result=audit(package)
            self.assertFalse(result['syntax_errors'])
            self.assertFalse(result['missing_entrypoints'])

if __name__=='__main__': unittest.main()
