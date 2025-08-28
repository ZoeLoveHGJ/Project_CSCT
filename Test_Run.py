# Test_Run.py
# -*- coding: utf-8 -*-
"""
ISC-BGCT 算法快速验证与调试脚本

*** V2.0 - 统一分析接口版 ***

该版本核心变更:
1.  【架构统一】: 彻底移除了脚本末尾的手动KPI计算逻辑。
2.  【接口调用】: 现在完全依赖 Tool.py 中的 SimulationAnalytics 类来处理
    原始仿真数据并计算所有性能指标。
3.  【结果一致性】: 确保了本测试脚本与 exp1.py 使用完全相同的
    数据分析流程，保证了结果的绝对可比性。

实验目的:
1. 在一个独立的、单线程的环境中运行一次ISC-BGCT算法，以验证其基本功能的正确性。
2. 通过详细的、逐步的日志输出，初步分析算法的运行时行为和潜在的性能瓶颈。
"""

import time

# 导入我们设计的框架、工具和算法实现
import Framework
import Tool
from algorithm_config import ALGORITHM_LIBRARY


# ==============================================================================
# 1. 测试场景与算法配置
# ==============================================================================

# 定义一个固定的、用于快速测试的场景
TEST_SCENARIO_CONFIG = {
    "simulation_duration_s": 3.0,
    "tag_arrival_rate_per_s": 5000,
    "mean_residence_time_ms": 1000.0,
    "std_dev_residence_time_ms": 200.0,
    "tag_id_length": 96,
}

# 直接指定要测试的算法及其配置
ALGO_NAME_TO_TEST = 'ISC-BGCT'
ALGO_INFO = ALGORITHM_LIBRARY[ALGO_NAME_TO_TEST]


# ==============================================================================
# 2. 带有详细日志的仿真主循环
# ==============================================================================

def run_verbose_simulation():
    """
    执行一次带有详细日志输出的单次仿真。
    """
    print("=" * 70)
    print(f"开始运行测试脚本: {__file__}")
    print(f"测试算法: {ALGO_NAME_TO_TEST}")
    print(f"测试场景: {TEST_SCENARIO_CONFIG}")
    print("=" * 70)
    
    # 1. 初始化
    scenario_config = Framework.StreamScenarioConfig(**TEST_SCENARIO_CONFIG)
    
    # 注意：这里我们不再需要手动管理 all_tags，因为 run_simulation 会返回它
    # algo_instance = ALGO_INFO["class"](ALGO_INFO["config"])
    
    start_time = time.time()

    # 2. 直接调用框架的 run_simulation 函数
    # 这确保了我们使用的是与 exp1.py 完全相同的仿真执行逻辑
    raw_results = Framework.run_simulation(
        scenario_config=scenario_config,
        algorithm_class=ALGO_INFO["class"],
        algorithm_specific_config=ALGO_INFO["config"]
        # 注意：这里没有传入 time_series_interval_ms，所以不会生成时序日志
    )

    # 3. 仿真结束，进行最终统计
    real_world_time_taken = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"仿真结束。真实世界耗时: {real_world_time_taken:.2f} 秒。")
    print("=" * 70)

    # --- 【核心修正】 ---
    # 4. 使用 Tool.py 进行数据分析和结果打印
    # 实例化分析工具
    analytics = Tool.SimulationAnalytics()

    # 将原始结果添加到分析器中，分析器内部会自动计算所有KPI
    analytics.add_run_result(
        raw_results=raw_results,
        scenario_config=TEST_SCENARIO_CONFIG,
        algorithm_name=ALGO_NAME_TO_TEST,
        run_id=0
    )

    # 从分析器获取包含所有计算好KPI的DataFrame
    results_df = analytics.get_results_dataframe()

    if results_df.empty:
        print("错误：未能从仿真结果中计算出任何性能指标。")
        return

    # 提取第一行（也是唯一一行）的结果进行打印
    final_kpis = results_df.iloc[0]

    # 打印性能摘要
    print("性能摘要 (由 Tool.py 计算):")
    print("-" * 30)
    
    # 从DataFrame中安全地获取值
    total_entered = int(final_kpis.get('system_throughput_tps', 0) * final_kpis.get('simulation_duration_s', 0) / (1 - final_kpis.get('tag_loss_rate', 1)))
    total_identified = int(final_kpis.get('system_throughput_tps', 0) * final_kpis.get('simulation_duration_s', 0))
    total_lost = total_entered - total_identified

    print(f"  进入视野的标签总数: {total_entered}")
    print(f"  成功识别的标签总数: {total_identified}")
    print(f"  丢失的标签总数:     {total_lost}")
    
    print(f"\n  标签识别率: {100 - final_kpis.get('tag_loss_rate', 0) * 100:.2f}%")
    print(f"  标签丢失率 (RLO): {final_kpis.get('tag_loss_rate', 0) * 100:.2f}%")
    print(f"  平均识别延迟 (DLY): {final_kpis.get('average_identification_delay_ms', 0):.2f} ms")
    print(f"  系统吞吐率 (THR): {final_kpis.get('system_throughput_tps', 0):.2f} tags/sec")
    print(f"  清场时间 (TCL): {final_kpis.get('time_to_clearance_ms', 0):.2f} ms")
    print(f"  协议繁忙度 (RBY): {final_kpis.get('ratio_of_busyness', 0) * 100:.2f}%")
    print("-" * 30)


# ==============================================================================
# 3. 脚本入口
# ==============================================================================

if __name__ == "__main__":
    run_verbose_simulation()
