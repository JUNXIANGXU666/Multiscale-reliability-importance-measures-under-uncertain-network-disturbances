"""Verify, analyse or recompute the released network-reliability experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from fetch_data import verify_data

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    manifest = json.loads((ROOT/'checksums.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = ROOT/row['path']
        if not path.is_file() or digest(path) != row['sha256']:
            raise RuntimeError('Missing or modified release file: '+row['path'])
    print(f"Verified {len(manifest['files'])} release files.", flush=True)


def copy_tree(source, target):
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


def prepare(work, mode):
    verify_data()
    marker = work/'run.json'
    if marker.exists():
        previous = json.loads(marker.read_text(encoding='utf-8'))
        if previous['mode'] != mode:
            raise ValueError('Use a different --work directory when changing reference/fresh mode.')
        print('Using existing '+mode+' workspace: '+str(work), flush=True)
        return
    if work.exists() and any(work.iterdir()):
        raise ValueError('The new work directory must be empty. Existing files are never deleted.')
    dpc, service = work/'dpc', work/'service'
    copy_tree(ROOT/'code/dpc', dpc)
    copy_tree(ROOT/'code/service', service/'experiments')
    copy_tree(ROOT/'config/service', service/'plans')
    copy_tree(ROOT/'data/sioux_falls', service/'experiments/inputs')
    (dpc/'sioux_inputs/SiouxFalls').mkdir(parents=True, exist_ok=True)
    for path in (ROOT/'data/sioux_falls').glob('*.tntp'):
        shutil.copyfile(path, dpc/'sioux_inputs/SiouxFalls'/path.name)
    for sub in ('audit', 'build', 'manuscript/figures', 'experiments/results', 'experiments/legacy_results'):
        (service/sub).mkdir(parents=True, exist_ok=True)
    if mode == 'reference':
        for source, target in [('baseline', 'results'), ('sioux', 'sioux_outputs/results'), ('robustness', 'robustness_outputs/results')]:
            copy_tree(ROOT/'reference/dpc'/source, dpc/target)
        for source in (ROOT/'reference/service').iterdir():
            if source.is_file():
                destination = service/'audit' if source.name == 'batched_propagation_equivalence.json' else service/'experiments/results'
                shutil.copyfile(source, destination/source.name)
        for name in ('grid_candidate_results.csv', 'grid_experiment_arrays.npz'):
            shutil.copyfile(dpc/'results'/name, service/'experiments/legacy_results'/name)
    marker.write_text(json.dumps({'mode': mode, 'source_manifest': digest(ROOT/'checksums.json')}, indent=2)+'\n', encoding='utf-8')
    print('Prepared '+mode+' workspace: '+str(work), flush=True)


def run(work, relative, *arguments):
    source = work/relative
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(work/'plot_config'), OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
    print('Running '+relative+' '+' '.join(arguments), flush=True)
    subprocess.run([sys.executable, '-B', '-X', 'utf8', str(source), *arguments],
                   cwd=source.parent, env=env, check=True)


def tests(work):
    run(work, 'service/experiments/service_validation.py', 'test')
    run(work, 'service/experiments/validate_batched_propagation.py')
    run(work, 'service/experiments/test_service_analysis.py')
    subprocess.run([sys.executable, '-B', str(ROOT/'tests/test_core.py'), str(work/'dpc')], check=True)


def analyse(work):
    # Recompute summaries from final scenario arrays, not from rounded paper tables.
    results = work/'service/experiments/results'
    required = ('primary_costs.npy', 'primary_features.npy', 'allocation_costs.npy',
                'allocation_features.npy', 'direction_costs.npy', 'direction_features.npy')
    missing = [name for name in required if not (results/name).is_file()]
    if missing:
        raise RuntimeError('Run the full experiments first. Scenario arrays are generated locally, '
                           'not distributed in the repository. Missing: '+', '.join(missing))
    completion = work/'service/experiments/results/service_analysis_completion.json'
    if completion.exists():
        completion.unlink()
    run(work, 'service/experiments/analyse_service_validation.py')
    run(work, 'dpc/summarise_interval_and_resampling.py')
    run(work, 'dpc/summarise_kernel_contrasts.py')


def render(work):
    subprocess.run([sys.executable, '-B', str(ROOT/'render_figures.py'), str(work)], check=True)


def all_experiments(work):
    run(work, 'dpc/run_all_experiments.py', '--phase', 'all')
    run(work, 'dpc/sioux_and_sensitivity.py', '--phase', 'all')
    run(work, 'dpc/propagation_scale_audit.py', '--stage', 'all')
    run(work, 'dpc/check_horizon_results.py')
    run(work, 'dpc/kernel_dependence_experiments.py', '--module', 'all')
    service_experiments(work)
    analyse(work)
    render(work)


def service_experiments(work):
    run(work, 'service/experiments/service_validation.py', 'test')
    run(work, 'service/experiments/confirmatory_validation.py', 'calibrate')
    run(work, 'service/experiments/service_coefficient_validation.py')
    run(work, 'service/experiments/validate_batched_propagation.py')
    run(work, 'service/experiments/heldout_validation.py', 'all')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'prepare', 'test', 'analyse', 'figures', 'service', 'all'])
    parser.add_argument('--mode', choices=['reference', 'fresh'], default='reference')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--confirm-long-run', action='store_true')
    args = parser.parse_args()
    verify()
    if args.action == 'verify':
        return
    work = (args.work or ROOT/'runs'/args.mode).resolve()
    if work == ROOT or ROOT.is_relative_to(work):
        raise ValueError('The work directory must not be the repository root or its ancestor.')
    if args.action in ('all', 'service') and not args.confirm_long_run:
        parser.error('Full numerical runs require --confirm-long-run and may take many hours.')
    if args.action in ('all', 'service', 'analyse') and args.mode != 'fresh':
        parser.error('Use --mode fresh. Reference files support tests and plots, not scenario-level reanalysis.')
    prepare(work, args.mode)
    actions = {'prepare': lambda _: None, 'test': tests, 'analyse': analyse, 'figures': render,
               'service': service_experiments, 'all': all_experiments}
    actions[args.action](work)


if __name__ == '__main__':
    main()
