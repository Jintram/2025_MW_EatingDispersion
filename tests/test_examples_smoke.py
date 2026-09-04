"""
Written by Claude, and not human-checked.

Smoke test: runs the three example scripts end-to-end and checks that they
finish without raising, and that they produce the output files they promise.
No numerical output is checked, only that the pipeline runs.

Each script is run as a subprocess in a temporary working directory that
mirrors the repository layout (input folders are symlinked in). Because the
example scripts use paths relative to the working directory, this means their
output lands in the temporary folder, so running this test does *not*
overwrite whatever is currently in Example_data/OUTPUT-* or
Synthetic_data/OUTPUT*.

Run from the root of the repository:
    python tests/test_examples_smoke.py
or:
    pytest tests/test_examples_smoke.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per example script: the input folders it needs (relative to the repo root),
# and a few output files it should have written when it is done.
EXAMPLES = {
    'leafstats_example_3channels.py': {
        'inputs': ['Example_data/DATA'],
        'expected_outputs': [
            'Example_data/OUTPUT-3channels/data_leaf_damage_singlemetrics.csv',
            'Example_data/OUTPUT-3channels/data_leaf_damage_singlemetrics.xlsx',
            'Example_data/OUTPUT-3channels/plots/Radial_acf.png',
            'Example_data/OUTPUT-3channels/plots/radial_pdfs.png',
        ],
    },
    'leafstats_example_1channel.py': {
        'inputs': ['Example_data/DATA'],
        'expected_outputs': [
            'Example_data/OUTPUT-1channel/data_leaf_damage_singlemetrics.csv',
            'Example_data/OUTPUT-1channel/data_leaf_damage_singlemetrics.xlsx',
            'Example_data/OUTPUT-1channel/plots/Radial_acf.png',
            'Example_data/OUTPUT-1channel/plots/radial_pdfs.png',
        ],
    },
    'leafstats_syntheticdata.py': {
        'inputs': ['Synthetic_data/images'],
        'expected_outputs': [
            'Synthetic_data/OUTPUT2/data_leaf_damage_singlemetrics.csv',
            'Synthetic_data/OUTPUT2/data_leaf_damage_singlemetrics.xlsx',
            'Synthetic_data/OUTPUT2/plots/Radial_acf.png',
            'Synthetic_data/OUTPUT2/plots/radial_pdfs.png',
        ],
    },
}


def _prepare_workdir(workdir, input_dirs):
    """Symlink the required input folders into the temporary working directory."""
    for rel_path in input_dirs:
        link_path = os.path.join(workdir, rel_path)
        os.makedirs(os.path.dirname(link_path), exist_ok=True)
        os.symlink(os.path.join(REPO_ROOT, rel_path), link_path)


def run_example(script_name):
    """Run one example script in a scratch working directory; return its outputs."""
    spec = EXAMPLES[script_name]

    workdir = tempfile.mkdtemp(prefix='leafstats_smoke_')
    try:
        _prepare_workdir(workdir, spec['inputs'])

        env = dict(os.environ)
        env['MPLBACKEND'] = 'Agg'          # no interactive windows
        env['PYTHONPATH'] = REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')

        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, script_name)],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"{script_name} exited with code {result.returncode}\n"
            f"--- stdout ---\n{result.stdout[-3000:]}\n"
            f"--- stderr ---\n{result.stderr[-3000:]}"
        )

        for rel_path in spec['expected_outputs']:
            out_path = os.path.join(workdir, rel_path)
            assert os.path.isfile(out_path), \
                f"{script_name} did not produce {rel_path}"
            assert os.path.getsize(out_path) > 0, \
                f"{script_name} produced an empty {rel_path}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_example_3channels_runs():
    run_example('leafstats_example_3channels.py')


def test_example_1channel_runs():
    run_example('leafstats_example_1channel.py')


def test_example_syntheticdata_runs():
    run_example('leafstats_syntheticdata.py')


if __name__ == '__main__':
    failures = 0
    for name, func in sorted(list(globals().items())):
        if name.startswith('test_') and callable(func):
            try:
                func()
            except AssertionError as e:
                failures += 1
                print(f'FAILED: {name}\n{e}\n')
            else:
                print(f'PASSED: {name}')

    if failures:
        print(f'\n{failures} test(s) failed.')
        sys.exit(1)
    print('\nAll tests passed.')
