# Exp_Sup_3.py
# -*- coding: utf-8 -*-

import time
import heapq
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10


QUEUE_SIZES = [100, 500, 1000, 2000, 3000, 4000, 5000]

TRIALS = 1000

RFID_SLOT_DURATION_US = 457.5

THRESHOLD_PERCENTAGE = 0.01
THRESHOLD_VALUE_US = RFID_SLOT_DURATION_US * THRESHOLD_PERCENTAGE


def benchmark_heap_operations(n_items):
    heap = []
    for _ in range(n_items):
        heapq.heappush(heap, (random.random(), object()))
    
    start_time = time.perf_counter()
    for _ in range(TRIALS):
        heapq.heappush(heap, (random.random(), object()))
        heapq.heappop(heap)
    end_time = time.perf_counter()
    
    avg_op_time_s = (end_time - start_time) / TRIALS
    return avg_op_time_s * 1e6 

if __name__ == "__main__":
    print("开始微基准测试...")
    latencies = []
    for n in QUEUE_SIZES:
        latency = benchmark_heap_operations(n)
        latencies.append(latency)
    print("测试完成，正在生成图表...")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    bar_color = '#3465a4'    
    line_color = '#cc0000'   
    text_color = '#333333'   

    x_pos = np.arange(len(QUEUE_SIZES))
    bars = ax1.bar(
        x_pos, 
        latencies, 
        color=bar_color, 
        edgecolor='black', 
        linewidth=0.5,
        alpha=0.85, 
        label='Computational Latency (Heap Push+Pop)'
    )

    line = ax1.axhline(y=THRESHOLD_VALUE_US, color=line_color, linestyle='--', linewidth=1.5, 
                       label=f'1% of Physical Slot ({THRESHOLD_VALUE_US:.2f} $\\mu$s)')

    ax1.text(len(QUEUE_SIZES) - 0.5, THRESHOLD_VALUE_US + 0.2, 
             f'Physical Slot Threshold (1% $\\approx$ {THRESHOLD_VALUE_US:.2f} $\\mu$s)', 
             color=line_color, fontsize=14, ha='right', va='bottom', fontweight='bold')

    ax1.set_xlabel('Priority Queue Size ($N$)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Processing Latency ($\\mu$s)', fontweight='bold', color=text_color, fontsize=14)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(n) for n in QUEUE_SIZES])

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f} $\\mu$s',
                ha='center', va='bottom', fontsize=14, color=text_color)

    legend = ax1.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray',fontsize=14)
    
    ax1.grid(axis='y', linestyle=':', linewidth=0.5, color='gray', alpha=0.7)
    
    ax1.set_ylim(0, THRESHOLD_VALUE_US * 1.5) 

    plt.tight_layout()

    output_file = "Figure_R3_Micro_Benchmark.png"
    output_file_pdf = "Figure_R3_Micro_Benchmark.pdf"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    
    print("-" * 60)
    print(f"已生成图表:")
    print(f"  - {output_file} ")
    print(f"  - {output_file_pdf}")
    print("-" * 60)
    
    plt.show()