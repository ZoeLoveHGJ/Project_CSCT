# Exp_Sup_2_Sensitivity.py
# -*- coding: utf-8 -*-

import time
import os
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from tqdm import tqdm

import Framework
import Tool
from algorithm_config import ALGORITHM_LIBRARY

NUM_RUNS = 5

FIXED_SCENARIO = {
    "simulation_duration_s": 10.0,
    "tag_arrival_rate_per_s": 2500, 
    "mean_residence_time_ms": 2000.0,
    "std_dev_residence_time_ms": 500.0,
    "tag_id_length": 96,
    "tag_failure_probability": 0.0,
    "capture_threshold_db": 0.0,
    "traffic_pattern": 'NORMAL'
}

STEP_SIZES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

TARGET_ALGO_KEY = 'CSCT'


def run_sensitivity_worker(task_params: dict) -> dict:
    scenario_config_dict = task_params['scenario_config']
    step_size = task_params['step_size']
    run_id = task_params['run_id']
    
    base_algo_info = ALGORITHM_LIBRARY[TARGET_ALGO_KEY]
    algo_class = base_algo_info["class"]
    algo_specific_config = deepcopy(base_algo_info["config"])
    
    algo_specific_config['step_size_C'] = step_size

    scenario_config = Framework.StreamScenarioConfig(**scenario_config_dict)
    
    raw_result_package = Framework.run_simulation(
        scenario_config=scenario_config,
        algorithm_class=algo_class,
        algorithm_specific_config=algo_specific_config
    )
    
    result_scenario_info = scenario_config_dict.copy()
    result_scenario_info['step_size_C'] = step_size
    
    return {
        "raw_results": raw_result_package,
        "scenario_config": result_scenario_info, 
        "algo_name": f"CSCT (C={step_size})",
        "run_id": run_id,
        "step_size": step_size
    }

if __name__ == "__main__":
    start_time = time.time()
    analytics = Tool.SimulationAnalytics()

    tasks = []
    for step in STEP_SIZES:
        for i in range(NUM_RUNS):
            tasks.append({
                "scenario_config": FIXED_SCENARIO,
                "step_size": step,
                "run_id": i
            })

    print(f"补充实验二: 参数 C 敏感性分析")
    print(f"固定负载: {FIXED_SCENARIO['tag_arrival_rate_per_s']} tags/s")
    print(f"测试范围: {STEP_SIZES}")
    print("-" * 60)

    with Pool(processes=min(len(tasks), cpu_count())) as pool:
        pbar = tqdm(pool.imap_unordered(run_sensitivity_worker, tasks), total=len(tasks), unit="run")
        
        for res in pbar:
            analytics.add_run_result(
                raw_results=res["raw_results"],
                scenario_config=res["scenario_config"],
                algorithm_name="CSCT_Sensitivity", 
                run_id=res["run_id"]
            )

    print("\n" + "-" * 60)
    print("正在生成敏感性分析图表...")
    
    output_dir = "exp_sup_2_sensitivity"
    
    analytics.save_to_csv(
        x_axis_key='step_size_C', 
        output_dir=output_dir
    )

    custom_style = {
        "CSCT_Sensitivity": {
            "label": "CSCT Performance",
            "style": {"color": "#d62728", "marker": "s", "linestyle": "-"}
        }
    }
    
    analytics.plot_results(
        x_axis_key='step_size_C', 
        algorithm_styles=custom_style,
        save_path=os.path.join(output_dir, "sensitivity_analysis.png")
    )

    end_time = time.time()
    print("-" * 60)
    print(f"实验完成。结果已保存至: {output_dir}")