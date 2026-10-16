"""Draw the relationship between the transport service model and DPC."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT/'build/mplconfig')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    font = font_manager.findfont('Arial', fallback_to_default=True)
    family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.set(xlim=(-7, 7), ylim=(-2.6, 6.1))
    ax.axis('off')
    blue, green, grey = '#245D8C', '#27735B', '#515B65'

    def box(x, y, width, height, text, edge, fill):
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
            boxstyle='round,pad=0.08,rounding_size=0.10', linewidth=.9,
            edgecolor=edge, facecolor=fill))
        ax.text(x, y, text, ha='center', va='center', linespacing=1.35)

    def arrow(start, end, colour=grey, style='-'):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=9,
            linewidth=.9, color=colour, linestyle=style))

    box(0, 5.25, 8.4, .9, 'Road attributes, OD demand\nand uncertain slowdowns', grey, '#F2F4F6')
    box(-3.65, 3.65, 5.5, .8, 'Directed travel times', green, '#EFF7F3')
    box(3.65, 3.65, 5.5, .8, 'CTQW amplitudes and occupations', blue, '#EFF5FA')
    box(-3.65, 1.85, 5.5, .9, 'Acceptable-service event', green, '#EFF7F3')
    box(3.65, 1.85, 5.5, .9, 'DPC and fitted threshold', blue, '#EFF5FA')
    box(0, -.05, 10.5, 1.0,
        'Baseline and intervention probabilities\nEdges, corridors and regions', grey, '#F2F4F6')
    box(0, -1.85, 12.2, 1.05,
        'Compare service and DPC importance\nApproximation, sampling and event-disagreement errors', grey, '#F2F4F6')
    arrow((-2.2, 4.72), (-3.65, 4.13))
    arrow((2.2, 4.72), (3.65, 4.13))
    arrow((-3.65, 3.17), (-3.65, 2.38), green)
    arrow((3.65, 3.17), (3.65, 2.38), blue)
    arrow((-.82, 1.85), (.82, 1.85), grey, '--')
    ax.text(0, 2.57, 'Training and\ncalibration', ha='center', va='center', fontsize=8.5,
            linespacing=1.15, color=grey)
    arrow((-3.65, 1.32), (-3.65, .53), green)
    arrow((3.65, 1.32), (3.65, .53), blue)
    arrow((0, -.63), (0, -1.24))
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    folder = ROOT/'manuscript/figures'
    folder.mkdir(parents=True, exist_ok=True)
    pdf = folder/'service_framework.pdf'
    png = ROOT/'build/service_framework.png'
    fig.savefig(pdf)
    fig.savefig(png, dpi=400)
    plt.close(fig)
    record = {'font_file': font, 'question': 'How do the independently defined service event and DPC enter one importance assessment?',
              'panels': 1, 'panel_reason': 'One shared input branches into two measurements and joins in paired comparisons.',
              'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
              'files': [str(pdf.relative_to(ROOT)), str(png.relative_to(ROOT))],
              'visual_status': 'pending'}
    (ROOT/'audit/framework_figure.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
