# -*- coding: utf-8 -*-
# language: english
"""
TAD: An Adapted RFID Anti-collision Algorithm in a Dynamic Environment
论文 "An Adapted RFID Anti-collision Algorithm in a Dynamic Environment" 的复现。
(W. Zhu, M. Li, J. Cao and X. Cui, WCNC 2020)

*** V1.1 - Bug Fix Version ***

======================================================================
核心修正:
1.  【BUG修复】: 修复了在识别阶段成功识别一个标签时 (num_responses == 1)，
    只设置了 `tag.is_identified = True`，但忘记计算并设置
    `tag.identification_time_us` 的错误。这个疏漏导致在数据后处理计算
    延迟时，会因尝试对 NoneType 进行数学运算而引发 TypeError。
2.  【依赖导入】: 为此，从 Framework 中导入了 calculate_time_delta 函数。
======================================================================
自定义参数说明:
- changing_ratio (float, 默认值: 0.2155):
  【源于论文】这是切换 PRB 和 ABS 策略的逗留率阈值。该值由原论文通过
  理论分析和仿真得出，是其算法的核心参数。
"""

# 1. 完整依赖和库导入
import math
from collections import deque
from typing import Dict, List, Optional, Set

# 导入仿真框架的核心组件
# 【关键修正】: 增加对 calculate_time_delta 的导入
from Framework import Tag, ReliableAlgorithmBase, AlgorithmStepResult, CONSTANTS, calculate_time_delta
from Tool import RfidUtils

class TAD_Algorithm(ReliableAlgorithmBase):
    def __init__(self, algorithm_specific_config: Dict):
        super().__init__(algorithm_specific_config)
        
        # --- 算法核心状态变量 ---
        self.phase = 'INIT'  # INIT, ESTIMATE_ARRIVING, ESTIMATE_STAYING, IDENTIFY

        # --- 标签估算相关状态 ---
        self.estimation_i = 1 # LoF 估算中的计数器
        self.estimated_arriving = 0
        self.estimated_staying = 0
        self.method_to_use = 'ABS' # 默认使用 ABS

        # --- 识别相关状态 ---
        self.prefix_stack: deque[str] = deque()
        self.tags_to_identify_in_round: Set[Tag] = set()

        # --- 历史数据 ---
        self.tags_in_last_frame: Set[Tag] = set()
        
        # --- 自定义参数 ---
        self.CHANGING_RATIO = algorithm_specific_config.get('changing_ratio', 0.2155)

    def get_backlog_size(self) -> int:
        """中文注释: 对于TAD，待办任务数是其内部识别堆栈的大小。"""
        return len(self.prefix_stack)

    def _start_new_round(self, active_tags: List[Tag]):
        """中文注释: 开启新一轮，重置状态并准备估算。"""
        self.phase = 'ESTIMATE_ARRIVING'
        self.estimation_i = 1
        self.tags_to_identify_in_round = {t for t in active_tags if not t.is_identified}

    def _get_rightmost_one_pos(self, tag_id: str) -> int:
        """中文注释: 辅助函数，用于LoF估算，找到ID中最右边'1'的位置 (从1开始计数)。"""
        try:
            return len(tag_id) - tag_id.rindex('1')
        except ValueError:
            return -1 # ID中没有'1'

    def execute_step(self, active_tags: List[Tag], current_time_us: float) -> Optional[AlgorithmStepResult]:
        """TAD 算法的核心状态机。"""
        if self.phase == 'INIT':
            self._start_new_round(active_tags)

        # --- 1. 到达标签估算阶段 ---
        if self.phase == 'ESTIMATE_ARRIVING':
            arriving_tags = self.tags_to_identify_in_round - self.tags_in_last_frame
            responding_tags = [t for t in arriving_tags if self._get_rightmost_one_pos(t.id) == self.estimation_i]
            
            # 【在这里插入新逻辑 - 到达估算阶段】
            responding_tags = self.get_actual_responders(responding_tags)

            if not responding_tags:
                self.estimated_arriving = 1.2897 * (2**(self.estimation_i - 1))
                self.phase = 'ESTIMATE_STAYING'
                self.estimation_i = 1
                return AlgorithmStepResult('internal_op', operation_description="TAD: Finished arriving estimation.")
            
            self.estimation_i += 1
            return AlgorithmStepResult('collision_slot' if len(responding_tags) > 1 else 'success_slot',
                                       reader_bits=CONSTANTS.QUERY_CMD_BITS,
                                       tag_bits=len(responding_tags) * 1,
                                       expected_max_tag_bits=1)

        # --- 2. 逗留标签估算与策略决策阶段 ---
        if self.phase == 'ESTIMATE_STAYING':
            staying_tags = self.tags_to_identify_in_round.intersection(self.tags_in_last_frame)

             # 【在这里插入新逻辑 - 逗留估算阶段】
            responding_tags = self.get_actual_responders(staying_tags)
            
            pre_tag_num = len(self.tags_in_last_frame)

            if pre_tag_num > 0:
                threshold = math.log2((pre_tag_num * self.CHANGING_RATIO) / 1.2897) + 1 if (pre_tag_num * self.CHANGING_RATIO) > 0 else float('inf')
                if self.estimation_i > threshold:
                    self.method_to_use = 'PRB'
                    self.phase = 'IDENTIFY'
                    self.prefix_stack.clear()
                    self.prefix_stack.append('')
                    return AlgorithmStepResult('internal_op', operation_description="TAD: Early termination, use PRB.")

            responding_tags = [t for t in staying_tags if self._get_rightmost_one_pos(t.id) == self.estimation_i]

            if not responding_tags:
                self.estimated_staying = 1.2897 * (2**(self.estimation_i - 1))
                staying_ratio = self.estimated_staying / pre_tag_num if pre_tag_num > 0 else 0
                self.method_to_use = 'PRB' if staying_ratio > self.CHANGING_RATIO else 'ABS'
                self.phase = 'IDENTIFY'
                self.prefix_stack.clear()
                self.prefix_stack.append('')
                return AlgorithmStepResult('internal_op', operation_description=f"TAD: Finished staying estimation, use {self.method_to_use}.")

            self.estimation_i += 1
            return AlgorithmStepResult('collision_slot' if len(responding_tags) > 1 else 'success_slot',
                                       reader_bits=CONSTANTS.QUERY_CMD_BITS,
                                       tag_bits=len(responding_tags) * 1,
                                       expected_max_tag_bits=1)

        # --- 3. 识别阶段 (ABS/PRB的统一实现) ---
        if self.phase == 'IDENTIFY':
            if not self.prefix_stack:
                self.tags_in_last_frame = self.tags_to_identify_in_round
                self.phase = 'INIT'
                return AlgorithmStepResult('internal_op', operation_description="TAD: Round finished.")

            query_prefix = self.prefix_stack.pop()
            
            responding_tags = [t for t in self.tags_to_identify_in_round if not t.is_identified and t.id.startswith(query_prefix)]

              # 【在这里插入新逻辑 - 识别阶段】
            responding_tags = self.get_actual_responders(responding_tags)

            num_responses = len(responding_tags)
            
            reader_bits = CONSTANTS.QUERY_CMD_BITS + len(query_prefix)

            if num_responses == 0:
                return AlgorithmStepResult('idle_slot', reader_bits=reader_bits)
            
            remaining_bits_len = CONSTANTS.EPC_CODE_BITS - len(query_prefix)
            
            if num_responses == 1:
                tag = responding_tags[0]
                
                result = AlgorithmStepResult('success_slot', reader_bits=reader_bits,
                                           tag_bits=remaining_bits_len, expected_max_tag_bits=remaining_bits_len)
                
                # --- 【关键修正】 ---
                # 计算本次成功识别所花费的时间，并为标签设置正确的识别时间戳。
                time_delta = calculate_time_delta(result)
                tag.identification_time_us = current_time_us + time_delta
                # --- 【修正结束】 ---

                tag.is_identified = True
                self.identified_tags_count += 1
                return result
            else: # 碰撞
                total_tag_bits = num_responses * remaining_bits_len
                
                # 标准二进制树分裂
                lcp = RfidUtils.get_collision_info([t.id for t in responding_tags])[0]
                
                # 论文的ABS/PRB是后进先出，所以先压'1'再压'0'
                self.prefix_stack.append(lcp + '1')
                self.prefix_stack.append(lcp + '0')
                
                return AlgorithmStepResult('collision_slot', reader_bits=reader_bits,
                                           tag_bits=total_tag_bits, expected_max_tag_bits=remaining_bits_len)

        return None
