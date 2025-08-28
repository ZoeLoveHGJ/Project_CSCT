
import random
import itertools
import pandas as pd
from typing import Dict, List, Any
import os
import concurrent.futures
from tqdm import tqdm

try:
    from Framework import StreamScenarioConfig, run_simulation
    from ISCT_Update3 import ISCT_Update3
    from Tool import SimulationAnalytics
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure Framework.py, ISCT_Update3.py, and Tool.py are in the same directory.")
    exit()

FIXED_SCENARIO_CONFIG = StreamScenarioConfig(
    simulation_duration_s=10.0,
    tag_arrival_rate_per_s=1000,
    mean_residence_time_ms=1000.0,
    std_dev_residence_time_ms=50.0
)

def run_single_experiment(params: Dict[str, Any]) -> Dict[str, float]:
    algo_config = {
        'initial_q': params['initial_q'],
        'priority_weight_size': params['priority_weight_size'],
        'priority_weight_depth': params['priority_weight_depth']
    }
    raw_results = run_simulation(
        scenario_config=FIXED_SCENARIO_CONFIG,
        algorithm_class=ISCT_Update3,
        algorithm_specific_config=algo_config,
        time_series_interval_ms=None
    )
    analytics = SimulationAnalytics()
    scenario_dict = {
        'simulation_duration_s': FIXED_SCENARIO_CONFIG.simulation_duration_s,
        'tag_arrival_rate_per_s': FIXED_SCENARIO_CONFIG.tag_arrival_rate_per_s
    }
    analytics.add_run_result(raw_results, scenario_dict, "ISCT_Update3", 0)
    results_df = analytics.get_results_dataframe()
    if not results_df.empty:
        return {
            'system_throughput_tps': results_df.iloc[0]['system_throughput_tps'],
            'tag_loss_rate': results_df.iloc[0]['tag_loss_rate'],
            'average_identification_delay_ms': results_df.iloc[0]['average_identification_delay_ms']
        }
    else:
        return {'system_throughput_tps': 0.0, 'tag_loss_rate': 1.0, 'average_identification_delay_ms': float('inf')}

if __name__ == "__main__":
    
    output_filename = "exp0_tuning_results_all.csv"
    all_results = []
    
    num_workers = os.cpu_count()
    print(f"Detected {num_workers} CPU cores. Using {num_workers} processes for parallel execution.")

    print("\n" + "="*60 + "\n### Stage 1: Starting Broad Random Search... ###\n" + "="*60)
    SEARCH_SPACE_RANDOM = {
        'initial_q': [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        'priority_weight_size': [0.1, 0.4, 0.5, 0.6, 1.0, 1.5, 2.0, 5.0],
        'priority_weight_depth': [0.05, 0.08, 0.1, 0.12, 0.2, 0.5, 1.0]
    }
    NUM_RANDOM_TRIALS = 50
    stage1_params_list = [{'initial_q': random.choice(SEARCH_SPACE_RANDOM['initial_q']), 'priority_weight_size': random.choice(SEARCH_SPACE_RANDOM['priority_weight_size']), 'priority_weight_depth': random.choice(SEARCH_SPACE_RANDOM['priority_weight_depth'])} for _ in range(NUM_RANDOM_TRIALS)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results_iterator = executor.map(run_single_experiment, stage1_params_list)
        stage1_kpis_results = list(tqdm(results_iterator, total=len(stage1_params_list), desc="[Stage 1] Random Search"))
        
    for params, kpis in zip(stage1_params_list, stage1_kpis_results):
        all_results.append({'stage': 1, **params, **kpis})

    stage1_results_df = pd.DataFrame([res for res in all_results if res['stage'] == 1])
    best_stage1_result = stage1_results_df.loc[stage1_results_df['system_throughput_tps'].idxmax()]
    
    print("\n" + "-"*60 + "\n### Stage 1 Complete! Best Result: ###")
    print(f"Parameters: { {k: v for k, v in best_stage1_result.items() if k in SEARCH_SPACE_RANDOM} }")
    print(f"Performance: Throughput={best_stage1_result['system_throughput_tps']:.2f}, Loss Rate={best_stage1_result['tag_loss_rate']:.3f}\n" + "-"*60 + "\n")

    print("="*60 + "\n### Stage 2: Starting Fine-grained Grid Search... ###\n" + "="*60)
    best_q = best_stage1_result['initial_q']; best_ws = best_stage1_result['priority_weight_size']; best_wd = best_stage1_result['priority_weight_depth']
    SEARCH_SPACE_GRID = {
        'initial_q': sorted(list(set([best_q - 1.0, best_q, best_q + 1.0]))),
        'priority_weight_size': sorted(list(set([round(best_ws * 0.8, 4), best_ws, round(best_ws * 1.2, 4)]))),
        'priority_weight_depth': sorted(list(set([round(best_wd * 0.8, 4), best_wd, round(best_wd * 1.2, 4)])))
    }
    grid_combinations = list(itertools.product(SEARCH_SPACE_GRID['initial_q'], SEARCH_SPACE_GRID['priority_weight_size'], SEARCH_SPACE_GRID['priority_weight_depth']))
    stage2_params_list = [{'initial_q': q, 'priority_weight_size': ws, 'priority_weight_depth': wd} for q, ws, wd in grid_combinations]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results_iterator = executor.map(run_single_experiment, stage2_params_list)
        stage2_kpis_results = list(tqdm(results_iterator, total=len(stage2_params_list), desc="[Stage 2] Grid Search"))
        
    for params, kpis in zip(stage2_params_list, stage2_kpis_results):
        all_results.append({'stage': 2, **params, **kpis})

    print("\n" + "="*60 + "\n### Simulation Run Complete! ###\n" + "="*60)
    results_df = pd.DataFrame(all_results)
    best_overall_result = results_df.loc[results_df['system_throughput_tps'].idxmax()]

    print("\n[Overall Best Parameter Set]:")
    best_params = {k: v for k, v in best_overall_result.items() if k in SEARCH_SPACE_RANDOM}
    print(f"  - Parameters: {best_params}")
    print(f"  - Performance:")
    print(f"    - System Throughput: {best_overall_result['system_throughput_tps']:.4f} tps")
    print(f"    - Tag Loss Rate: {best_overall_result['tag_loss_rate']:.4f}")
    print(f"    - Average Delay: {best_overall_result['average_identification_delay_ms']:.4f} ms")

    results_df.to_csv(output_filename, index=False, float_format='%.4f')
    print(f"\nAll {len(all_results)} experiment results have been saved to: {output_filename}")