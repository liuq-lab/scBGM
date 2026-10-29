"""Thin, dependency-light dispatcher. Heavy libraries load only in child scripts."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from . import __version__
from .registry import COMMANDS, ROOT


def launch_plan(command, arguments):
    relative, helpers = COMMANDS[command]
    script = ROOT / relative
    paths = [script.parent, *(ROOT / p for p in helpers), ROOT / "models", ROOT.parent]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths)) + (
        os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    # Several original evaluators default to Linux /dev/shm; keep caches portable.
    for name in ('TMPDIR', 'XDG_CACHE_HOME', 'JOBLIB_TEMP_FOLDER'):
        env.setdefault(name, tempfile.gettempdir())
    return [sys.executable, str(script), *arguments], env


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--version']:
        print(__version__)
        return 0
    if argv[:1] == ['list']:
        print('setup\tCheck training environment and native scIB; --repair-scib compiles it')
        print('data hvg\tUnified TRAIN-only HVG selection (all datasets)')
        for command, (path, _) in COMMANDS.items():
            print(f'{command}\t{path}')
        return 0
    if argv[:1] == ['doctor']:
        from .audit import main as doctor
        return doctor(argv[1:])
    if argv[:1] == ['setup']:
        from .environment import main as check
        return check(argv[1:])
    if argv[:2] == ['data', 'hvg']:
        from .data.hvg import main as hvg
        return hvg(argv[2:])
    dry_run = argv[:1] == ['--dry-run']
    if dry_run:
        argv.pop(0)
    command = ' '.join(argv[:2])
    if command in COMMANDS:
        arguments = argv[2:]
        if arguments[:1] == ['--']:
            arguments = arguments[1:]
        cmd, env = launch_plan(command, arguments)
        if dry_run:
            print(json.dumps({'argv': cmd, 'PYTHONPATH': env['PYTHONPATH'],
                              'cwd': str(Path.cwd())}, indent=2))
            return 0
        return subprocess.run(cmd, env=env).returncode
    parser = argparse.ArgumentParser(prog='scbgm', description=(
        'Unified scBGM interface. Run `scbgm list` for commands. '
        'Use `scbgm data hvg --help` or `scbgm train conditional --help`. '
        'Workflow commands accept script-specific arguments. '
        '`scbgm --dry-run <group> <command> ...` previews dispatch.'))
    parser.add_argument('--version', action='version', version=__version__)
    if not argv or argv == ['--help'] or argv == ['-h']:
        parser.print_help()
        return 0
    parser.error(f'Unknown command: {shlex.join(argv)}. Run `scbgm list`.')
