"""Check syntax, registered scripts, placeholder input paths and dependencies."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
from .registry import ROOT, COMMANDS


def audit(root=ROOT):
    release = root.parent
    syntax_errors, placeholder_paths = [], []
    python_files = list(root.rglob('*.py'))
    for path in python_files:
        text = path.read_text()
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append({'path':str(path.relative_to(root)),'line':exc.lineno,'error':str(exc)})
        for lineno,line in enumerate(text.splitlines(),1):
            if '/path/to/your/' in line:
                placeholder_paths.append({'path':str(path.relative_to(root)), 'line':lineno,'text':line.strip()[:240]})
    missing_entrypoints = [name for name,(path,_) in COMMANDS.items() if not (root/path).is_file()]
    dependencies = {name:importlib.util.find_spec(name) is not None for name in
                    ('anndata','numpy','pandas','scipy','scanpy','skmisc','sklearn',
                     'tensorflow','tensorflow_probability','torch','scib')}
    return {'runtime_root':str(root), 'python_files':len(python_files),
            'syntax_errors':syntax_errors,'missing_entrypoints':missing_entrypoints,
            'placeholder_path_occurrences':len(placeholder_paths),'placeholder_paths':placeholder_paths,
            'dependencies_available':dependencies,
            'bayesgm_license_present':(root/'models/bayesgm/LICENSE').is_file(),
            'project_license_present':(release/'LICENSE').is_file(),
            'this_audit_executes_training_or_data':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',type=Path,help='Write the full audit as JSON')
    args = parser.parse_args(argv)
    result = audit()
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(result,indent=2)+'\n')
    print(f"Runtime Python files: {result['python_files']}; syntax errors: {len(result['syntax_errors'])}")
    print(f"Missing registered scripts: {len(result['missing_entrypoints'])}")
    print(f"Placeholder paths to set before running paper-specific scripts: {result['placeholder_path_occurrences']}")
    print('Unavailable dependencies: '+', '.join(k for k,v in result['dependencies_available'].items() if not v))
    print('Project license: '+('present' if result['project_license_present'] else 'MISSING (author decision required)'))
    return int(bool(result['syntax_errors'] or result['missing_entrypoints']))
