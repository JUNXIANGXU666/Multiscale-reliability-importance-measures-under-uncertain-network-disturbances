"""Render the archived grid results with the current DPC terminology."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT/'build/mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

DATA = ROOT/'experiments/legacy_results'
BLUE, CYAN, RED = '#3C5488', '#4DBBD5', '#E64B35'
LIGHT, DARK = '#D9D9D9', '#303030'
DIVERGING = LinearSegmentedColormap.from_list('dpc_diverging', [BLUE, '#F7F7F7', RED])
SEQUENTIAL = LinearSegmentedColormap.from_list('dpc_sequential', ['#F4F7FA', CYAN, BLUE])


def half_up(value):
    return str(Decimal(str(value)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))


def network(ax, edges, positions, colours, width):
    for k, (u, v) in enumerate(edges):
        ax.plot(positions[[u, v], 0], positions[[u, v], 1], color=colours[k],
                linewidth=width, solid_capstyle='round', zorder=1)
    ax.scatter(positions[:, 0], positions[:, 1], s=24, facecolor='white',
               edgecolor=DARK, linewidth=.7, zorder=3)
    ax.set_aspect('equal')
    ax.margins(.22)
    ax.set_axis_off()


def main():
    with np.load(DATA/'grid_experiment_arrays.npz') as data:
        edges, positions = data['edges'].astype(int), data['positions']
    assert len(edges) == 40 and positions.shape == (25, 2)
    candidates = pd.read_csv(DATA/'grid_candidate_results.csv')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 9,
                         'axes.labelsize': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.05), layout='constrained')
    values = np.zeros(len(edges))
    for row in candidates[candidates.scale == 'edge'].itertuples(index=False):
        values[int(row.candidate_id[1:])-1] = row.exact_importance
    limit = np.abs(values).max()
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    network(axes[0], edges, positions, DIVERGING(norm(values)), 3.4)
    axes[0].set_title('Edge DPC importance')
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=DIVERGING), ax=axes[0], shrink=.78, pad=.025)
    cb.set_label(r'DPC importance, $\mathcal{I}_L$')
    top = candidates[candidates.scale == 'corridor'].nlargest(1, 'exact_importance').iloc[0]
    selected = set(map(int, str(top.edge_ids).split()))
    network(axes[1], edges, positions, [LIGHT]*len(edges), 1.6)
    for k in selected:
        u, v = edges[k]
        axes[1].plot(positions[[u, v], 0], positions[[u, v], 1], color=RED,
                     linewidth=4.6, solid_capstyle='round', zorder=2)
    axes[1].set_title('Highest-ranked corridor, '+top.candidate_id)
    axes[1].text(.5, -.03, r'$\mathcal{I}_L='+half_up(top.exact_importance)+'$',
                 transform=axes[1].transAxes, ha='center', va='top')
    regions = candidates[candidates.scale == 'region']
    values = np.zeros(len(edges))
    for row in regions.itertuples(index=False):
        values[list(map(int, str(row.edge_ids).split()))] = row.exact_importance
    norm = Normalize(vmin=0, vmax=regions.exact_importance.max())
    network(axes[2], edges, positions, SEQUENTIAL(norm(values)), 3.6)
    for row, (x, y) in zip(regions.itertuples(index=False), [(1,3.25),(3,3.25),(1,.75),(3,.75)]):
        axes[2].text(x, y, row.candidate_id+'\n'+half_up(row.exact_importance),
                     ha='center', va='center', fontsize=7.5, color=DARK,
                     bbox={'boxstyle':'round,pad=.22','facecolor':'white','edgecolor':LIGHT,
                           'linewidth':.6,'alpha':.92}, zorder=5)
    axes[2].set_title('Regional DPC importance')
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=SEQUENTIAL), ax=axes[2], shrink=.78, pad=.025)
    cb.set_label(r'DPC importance, $\mathcal{I}_L$')
    for ax, letter in zip(axes, 'abc'):
        ax.text(-.03, 1.06, '('+letter+')', transform=ax.transAxes, va='top',
                fontsize=9, fontweight='bold', color=DARK)
    path = ROOT/'manuscript/Figure 4.pdf'
    fig.savefig(path, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    print('Rendered grid DPC importance: '+str(path))


if __name__ == '__main__':
    main()
