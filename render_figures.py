"""Regenerate the eleven manuscript figures from a prepared numerical workspace."""
import importlib
import os
from pathlib import Path
import shutil
import sys

work = Path(sys.argv[1]).resolve()
os.environ.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(work/'plot_config'))
import matplotlib.pyplot as plt

sys.path.insert(0, str(work/'dpc'))
with plt.rc_context():
    base = importlib.import_module('render_base_figures')
    base.configure_style()
    base.coefficient_validation_figure()
    base.robustness_ablation_figure()
with plt.rc_context():
    interval = importlib.import_module('render_interval_figure')
    interval.certificate_map_figure()
with plt.rc_context():
    comparisons = importlib.import_module('render_dpc_comparisons')
    comparisons.quadratic_approximation_calibration()
    comparisons.kernel_comparison()
    comparisons.sioux_agreement()
    comparisons.runtime_scaling()

sys.path.insert(0, str(work/'service/experiments'))
with plt.rc_context():
    importlib.import_module('render_framework').main()
with plt.rc_context():
    importlib.import_module('render_service_results').main()
grid_data = work/'service/experiments/legacy_results'
grid_data.mkdir(parents=True, exist_ok=True)
for name in ('grid_experiment_arrays.npz', 'grid_candidate_results.csv'):
    shutil.copyfile(work/'dpc/results'/name, grid_data/name)
with plt.rc_context():
    importlib.import_module('render_grid_dpc').main()

mapping = {
    1: 'service/manuscript/figures/service_framework.pdf',
    2: 'service/manuscript/figures/service_importance_comparison.pdf',
    3: 'service/manuscript/figures/service_horizon_sensitivity.pdf',
    4: 'dpc/figures/coefficient_validation.pdf',
    5: 'dpc/figures_r1/certificate_operating_map.pdf',
    6: 'service/manuscript/Figure 4.pdf',
    7: 'dpc/robustness_outputs/figures/quadratic_approximation_calibration.pdf',
    8: 'dpc/robustness_outputs/figures/kernel_comparison.pdf',
    9: 'dpc/figures/topology_robustness_and_ablation.pdf',
    10: 'dpc/robustness_outputs/figures/sioux_benchmark_comparison_r2.pdf',
    11: 'dpc/robustness_outputs/figures/interaction_cost_and_storage_r2.pdf',
}
output = work/'figures'
output.mkdir(parents=True, exist_ok=True)
for number, source in mapping.items():
    shutil.copyfile(work/source, output/f'Figure_{number:02d}.pdf')
plt.close('all')
print('Regenerated eleven figures in '+str(output), flush=True)
