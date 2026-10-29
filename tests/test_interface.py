import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from scbgm.cli import main, launch_plan
from scbgm.registry import ROOT, COMMANDS
from scbgm.audit import audit

class InterfaceTests(unittest.TestCase):
    def test_snapshot_integrity(self):
        result = audit()
        self.assertFalse(result['syntax_errors'])
        self.assertFalse(result['missing_entrypoints'])

    def test_local_helpers_are_unambiguous(self):
        expected = {'train unconditional': 'training/unconditional', 'train conditional': 'models/autoencoders/conditional',
                    'train trajectory': 'models/autoencoders/conditional', 'train counterfactual': 'models/autoencoders/conditional',
                    'generate counterfactual': 'models/autoencoders/conditional'}
        for command, folder in expected.items():
            cmd, env = launch_plan(command, ['--help'])
            search = [Path(cmd[1]).parent, *map(Path, env['PYTHONPATH'].split(os.pathsep))]
            found = next(p/'autoencoder.py' for p in search if (p/'autoencoder.py').is_file())
            self.assertEqual(found.resolve(), (ROOT/folder/'autoencoder.py').resolve())

    def test_forwarding_and_exit_code_in_real_child(self):
        # Replace one registry entry with a harmless stub, then exercise the actual subprocess.
        with tempfile.TemporaryDirectory() as folder:
            stub = Path(folder)/'stub.py'
            stub.write_text('import sys\nassert sys.argv[1:] == ["--value", "a b;$HOME"]\nsys.exit(7)\n')
            COMMANDS['test stub'] = (str(stub), ())
            try:
                self.assertEqual(main(['test','stub','--','--value','a b;$HOME']), 7)
            finally:
                del COMMANDS['test stub']

    def test_dry_run_starts_no_process(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--dry-run','train','conditional','--output-dir','a b']), 0)
        self.assertIn('a b', output.getvalue())

    def test_data_help_without_scientific_imports(self):
        child = subprocess.run([sys.executable, '-c',
            'import sys; from scbgm.cli import main\n'
            'try: main(["data","hvg","--help"])\n'
            'except SystemExit as e: assert e.code == 0\n'
            'assert "scanpy" not in sys.modules\nassert "tensorflow" not in sys.modules'],
            capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_unknown_command_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            main(['train','typo'])
        self.assertEqual(exc.exception.code, 2)

if __name__ == '__main__':
    unittest.main()
