"""Produce publication tables and figures only from completed service analyses."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT/'build/mplconfig')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
import numpy as np
import pandas as pd

RESULTS = ROOT/'experiments/results'
FIGURES = ROOT/'manuscript/figures'
FORMS = {'edge': 'Edges', 'corridor': 'Corridors', 'region': 'Regions'}
METHODS = {
    'adjacency_1.00': 'Adjacency CTQW', 'laplacian_quantum': 'Laplacian CTQW',
    'classical_diffusion': 'Classical diffusion', 'local_retention': 'Local squared-coupling score',
    'late_triangular': 'Late-triangular window', 'instantaneous': 'Instantaneous observation',
    'first_order': 'First-order approximation', 'second_order': 'Second-order approximation'}


def number(value, digits=3):
    return '--' if pd.isna(value) else f'{value:.{digits}f}'


def table(label, caption, headings, widths, rows):
    columns = ''.join(r'>{\raggedright\arraybackslash}p{'+str(w)+r'\textwidth}' for w in widths)
    header = ' & '.join(headings)+r' \\'
    body = '\n'.join(' & '.join(map(str, row))+r' \\' for row in rows)
    return ('\\begingroup\n\\small\n\\singlespacing\n'
            '\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.12}\n'
            '\\begin{longtable}{'+columns+'}\n\\caption{'+caption+'}\\label{'+label+r'}\\'+'\n'
            '\\toprule\n'+header+'\n\\midrule\n\\endfirsthead\n\\toprule\n'+header+
            '\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n'+body+
            '\n\\end{longtable}\n\\endgroup\n\\FloatBarrier\n')


def verify_inputs():
    path = RESULTS/'service_analysis_completion.json'
    complete = json.loads(path.read_text(encoding='utf-8'))
    assert complete['status'] == 'complete'
    for item in complete['artifacts']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest() == item['sha256']
    return complete


def make_tables(summary, intervals, sensitivity, allocation, direction, calibration):
    tables = {}
    rows = []
    for key, label in [('adjacency_quantum', 'Adjacency CTQW'), ('laplacian_quantum', 'Laplacian CTQW'),
                       ('classical_diffusion', 'Classical diffusion')]:
        s = calibration['selected'][key]
        rows.append([label, number(s['horizon'], 1), number(s['threshold'], 5),
                     number(s['balanced_error'], 4), number(s['max_local_balanced_error'], 4),
                     'Yes' if s['eligible'] else 'No'])
    tables['calibration'] = table('tab:service-calibration', 'Calibration against the travel-service requirement.',
        ['Kernel', r'\(L\)', r'\(\zeta\)', 'Balanced error', r'Largest \(\pm5\%\) error', 'Admissible'],
        [.22, .06, .12, .14, .18, .13], rows)

    primary = summary[np.isclose(summary.service_limit, 1.10)]
    rows = []
    main_methods = ['adjacency_1.00', 'laplacian_quantum', 'classical_diffusion', 'local_retention']
    for form, form_label in FORMS.items():
        for method in main_methods:
            row = primary[(primary.form == form) & (primary.method == method)].iloc[0]
            ci = intervals[(intervals.form == form) & (intervals.method == method)].iloc[0]
            rows.append([form_label, METHODS[method], number(row.balanced_error, 4),
                number(row.importance_mae, 4)+' ['+number(ci.mae_low, 4)+', '+number(ci.mae_high, 4)+']',
                number(row.spearman), number(row.top_five_overlap)])
    tables['importance'] = table('tab:service-importance', 'Held-out service classification and importance accuracy.',
        ['Form', 'Method', 'Balanced error', r'Importance MAE [95\% interval]', 'Rank correlation', 'Top-five overlap'],
        [.10, .22, .12, .25, .13, .10], rows)

    rows = []
    for method in ['adjacency_1.00', 'late_triangular', 'instantaneous']:
        for form, form_label in FORMS.items():
            row = primary[(primary.method == method) & (primary.form == form)].iloc[0]
            label = 'Uniform window' if method == 'adjacency_1.00' else METHODS[method]
            rows.append([label, form_label, number(row.balanced_error, 4), number(row.importance_mae, 4),
                         number(row.spearman), number(row.top_five_overlap)])
    tables['windows'] = table('tab:service-observation-window', 'Sensitivity to the observation window.',
        ['Measurement', 'Form', 'Balanced error', 'Importance MAE', 'Rank correlation', 'Top-five overlap'],
        [.24, .11, .14, .14, .14, .13], rows)

    rows = []
    for rule in ['relative', 'raw']:
        for error in [.01, .05, .10]:
            subset = sensitivity[np.isclose(abs(sensitivity.factor-1), error) & (sensitivity.rule == rule)]
            for form, label in FORMS.items():
                s = subset[subset.form == form]
                assert len(s) == 2
                rows.append(['Relative' if rule == 'relative' else 'Fixed raw', f'{100*error:.0f}'+r'\%', label,
                    '--' if s.spearman.isna().any() else number(s.spearman.min()), number(s.reversal_fraction.max()),
                    number(s.top_five_overlap.min()), str(int(s.sign_changes.max()))])
    tables['horizon'] = table('tab:service-small-scale-errors', 'Worst ranking changes under small observation-scale errors.',
        ['Threshold', 'Error', 'Form', 'Minimum rank correlation', 'Maximum reversal fraction',
         'Minimum top-five overlap', 'Maximum sign changes'], [.13, .07, .10, .16, .16, .17, .11], rows)

    rows = []
    for form, label in FORMS.items():
        for metric in ['service', 'DPC']:
            s = allocation[np.isclose(allocation.service_limit, 1.10) & (allocation.form == form) &
                           (allocation.metric == metric)].iloc[0]
            rows.append([label, 'Service' if metric == 'service' else 'DPC',
                number(s.mean_absolute_allocation_difference, 4), number(s.maximum_absolute_allocation_difference, 4),
                number(s.spearman), number(s.top_five_overlap)])
    tables['allocation'] = table('tab:service-allocation', 'Effect of allocation under the same realised total disturbance.',
        ['Form', 'Importance', 'Mean absolute difference', 'Maximum difference', 'Rank correlation', 'Top-five overlap'],
        [.12, .14, .19, .16, .15, .14], rows)

    rows = []
    for form, label in FORMS.items():
        s = direction[np.isclose(direction.service_limit, 1.10) & (direction.form == form)].iloc[0]
        rows.append([label, number(s.mean_relative_cost_bias*100, 2)+r'\%',
            number(s.event_disagreement, 4), number(s.symmetrisation_importance_mae, 4),
            number(s.maximum_symmetrisation_importance_error, 4), number(s.spearman), number(s.top_five_overlap)])
    tables['direction'] = table('tab:service-direction-bias', 'Service bias from reciprocal-rate symmetrisation.',
        ['Form', 'Mean cost bias', 'Event disagreement', 'Importance MAE', 'Maximum error', 'Rank correlation', 'Top-five overlap'],
        [.11, .13, .14, .13, .13, .13, .13], rows)

    rows = []
    for method in ['first_order', 'second_order']:
        for form, label in FORMS.items():
            s = primary[(primary.method == method) & (primary.form == form)].iloc[0]
            rows.append([METHODS[method], label, number(s.exact_dpc_importance_mae, 4),
                         number(s.importance_mae, 4), number(s.spearman)])
    tables['approximation'] = table('tab:service-approximation', 'Approximation error and service error under the calibrated setting.',
        ['Approximation', 'Form', 'MAE against exact DPC', 'MAE against service', 'Service rank correlation'],
        [.24, .12, .20, .18, .18], rows)

    rows = []
    for limit in [1.05, 1.20]:
        for form, label in FORMS.items():
            for method in main_methods:
                s = summary[np.isclose(summary.service_limit, limit) & (summary.method == method) & (summary.form == form)].iloc[0]
                rows.append([number(limit, 2), label, METHODS[method], number(s.balanced_error, 4),
                             number(s.importance_mae, 4), number(s.spearman), number(s.top_five_overlap)])
    tables['secondary'] = table('tab:service-secondary-thresholds', 'Results under secondary travel-service requirements.',
        [r'\(c\)', 'Form', 'Method', 'Balanced error', 'Importance MAE', 'Rank correlation', 'Top-five overlap'],
        [.06, .09, .22, .14, .14, .13, .13], rows)
    return tables


def save(fig, name):
    path = FIGURES/(name+'.pdf')
    fig.savefig(path)
    fig.savefig(ROOT/'build'/(name+'.png'), dpi=400)
    plt.close(fig)
    return path


def scatter_figure(candidates):
    frame = candidates[np.isclose(candidates.service_limit, 1.10)]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7))
    styles = [('adjacency_1.00', '#2869A6', 'o'), ('classical_diffusion', '#C15F25', '^')]
    for i, (ax, (form, label)) in enumerate(zip(axes, FORMS.items())):
        subset = frame[frame.form == form]
        x, y = [], []
        for method, colour, marker in styles:
            s = subset[subset.method == method]
            ax.scatter(s.service_importance, s.estimated_importance, s=8, c=colour, marker=marker,
                       alpha=.55, linewidths=0, rasterized=False)
            x.extend(s.service_importance)
            y.extend(s.estimated_importance)
        lo = min(0, min(x), min(y))
        hi = max(max(x), max(y))
        span = max(hi-lo, .02)
        limits = (lo-.04*span, hi+.04*span)
        ax.plot(limits, limits, '--', color='#777777', linewidth=.8, zorder=0)
        ax.set(xlim=limits, ylim=limits, xlabel='Service importance')
        ax.set_title(f'({chr(97+i)}) {label}', loc='left', fontsize=9)
        ax.grid(alpha=.17, linewidth=.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Estimated importance')
    handles = [Line2D([], [], color=c, marker=m, linestyle='None', markersize=4, label=METHODS[k])
               for k, c, m in styles]
    handles.append(Line2D([], [], color='#777777', linestyle='--', linewidth=.8, label='Exact agreement'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=.085, right=.99, bottom=.19, top=.78, wspace=.34)
    return save(fig, 'service_importance_comparison')


def horizon_figure(sensitivity):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7))
    styles = [('relative', '#2869A6', 'o', '-', 'Fixed relative threshold'),
              ('raw', '#C15F25', 's', '--', 'Fixed raw threshold')]
    finite = sensitivity[sensitivity.form.isin(FORMS) & np.isfinite(sensitivity.spearman)].spearman
    low = min(.8, float(finite.min())-.05)
    for i, (ax, (form, label)) in enumerate(zip(axes, FORMS.items())):
        for rule, colour, marker, line, legend in styles:
            s = sensitivity[(sensitivity.form == form) & (sensitivity.rule == rule)].sort_values('factor')
            ax.plot(s.factor, s.spearman, color=colour, marker=marker, markersize=3,
                    linewidth=1, linestyle=line, label=legend)
        ax.set(xlim=(.89, 1.11), ylim=(max(-1.05, low), 1.02), xlabel="Observation scale, L'/L")
        ax.set_title(f'({chr(97+i)}) {label}', loc='left', fontsize=9)
        ax.set_xticks([.90, .95, 1., 1.05, 1.10])
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.2, linewidth=.5)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Rank correlation')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=.085, right=.99, bottom=.19, top=.78, wspace=.34)
    return save(fig, 'service_horizon_sensitivity')


def main():
    complete = verify_inputs()
    font = font_manager.findfont('Arial', fallback_to_default=True)
    family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 9, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.labelsize': 8.5})
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(RESULTS/(name+'.csv')) for name in [
        'service_validation_summary', 'service_validation_intervals', 'horizon_sensitivity_summary',
        'allocation_comparison_summary', 'direction_bias_summary', 'service_validation_candidates']}
    calibration = json.loads((RESULTS/'calibration_completion.json').read_text(encoding='utf-8'))
    tables = make_tables(frames['service_validation_summary'], frames['service_validation_intervals'],
        frames['horizon_sensitivity_summary'], frames['allocation_comparison_summary'],
        frames['direction_bias_summary'], calibration)
    table_path = ROOT/'build/service_tables.json'
    table_path.write_text(json.dumps(tables, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    figures = [scatter_figure(frames['service_validation_candidates']),
               horizon_figure(frames['horizon_sensitivity_summary'])]
    artifacts = figures+[table_path]
    record = {'status': 'generated', 'analysis_fingerprint': complete['fingerprint'],
        'font_file': font, 'tables': list(tables),
        'figures': [{'name': 'service_importance_comparison', 'panels': 3,
                     'question': 'Do calibrated DPC and classical diffusion preserve independently computed service importance within each structural form?',
                     'rationale': 'The three forms share the same comparison, but have different numerical ranges and are reported separately.'},
                    {'name': 'service_horizon_sensitivity', 'panels': 3,
                     'question': 'How does a small scale error change rankings under the two threshold rules?',
                     'rationale': 'Aligned form-specific panels expose the same controlled scale change.'}],
        'artifacts': [{'path': str(p.relative_to(ROOT)).replace('\\', '/'),
                       'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts]}
    (ROOT/'audit/service_publication_assets.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
