import math
from collections import deque
from typing import Dict, List, Optional, Set


from Framework import Tag, ReliableAlgorithmBase, AlgorithmStepResult, CONSTANTS, calculate_time_delta
from Tool import RfidUtils


class ABP_Algorithm(ReliableAlgorithmBase):
    def __init__(self, algorithm_specific_config: Dict):
        super().__init__(algorithm_specific_config)

        self.phase = 'INIT'

        self.high_priority_tags: Set[Tag] = set()

        self.frame_length = algorithm_specific_config.get(
            'initial_frame_length', 128)
        self.slot_counter = 0
        self.frame_outcomes = {'idle': 0, 'success': 0, 'collision': 0}

    def get_backlog_size(self) -> int:
        """中文注释: 对于ABP，待办任务数是当前识别周期中还未被识别的高优先级标签数量。"""

        active_high_priority_tags = self.high_priority_tags.intersection(
            self.active_tags_snapshot)
        return len([t for t in active_high_priority_tags if not t.is_identified])

    def _start_new_cycle(self, active_tags: List[Tag]):
        """中文注释: 开启一个全新的识别周期，确定高优先级标签。"""

        self.high_priority_tags = {
            t for t in active_tags if not t.is_identified}

        if not self.high_priority_tags:
            self.phase = 'INIT'
            return

        self.phase = 'IDENTIFYING_HIGH_PRIORITY'

        self.frame_length = 128
        self.slot_counter = 0
        self.frame_outcomes = {'idle': 0, 'success': 0, 'collision': 0}

    def execute_step(self, active_tags: List[Tag], current_time_us: float) -> Optional[AlgorithmStepResult]:
        """ABP 算法的核心状态机。"""

        self.active_tags_snapshot = set(active_tags)

        if self.phase == 'INIT':
            self._start_new_cycle(active_tags)

            if not self.high_priority_tags:
                return None

        if self.phase == 'IDENTIFYING_HIGH_PRIORITY':

            active_high_priority_tags = self.high_priority_tags.intersection(
                self.active_tags_snapshot)
            unidentified_targets = [
                t for t in active_high_priority_tags if not t.is_identified]

            if not unidentified_targets:

                self.phase = 'INIT'

                self._start_new_cycle(active_tags)
                if not self.high_priority_tags:
                    return None

                return AlgorithmStepResult('internal_op', operation_description="ABP: Cycle finished, starting new one.")

            if self.slot_counter >= self.frame_length:

                mc = self.frame_outcomes['collision']

                estimated_remaining_tags = 2.39 * mc

                if estimated_remaining_tags > 0:

                    q_value = round(math.log2(estimated_remaining_tags))
                    self.frame_length = max(1, int(2**q_value))
                else:

                    self.frame_length = 1

                self.slot_counter = 0
                self.frame_outcomes = {'idle': 0, 'success': 0, 'collision': 0}

            current_slot = self.slot_counter

            responding_tags = [
                tag for tag in unidentified_targets
                if (int(tag.id, 2) % self.frame_length == current_slot)
            ]

            responding_tags = self.get_actual_responders(responding_tags)
            num_responses = len(responding_tags)

            if num_responses == 0:
                self.frame_outcomes['idle'] += 1
            elif num_responses == 1:
                self.frame_outcomes['success'] += 1
            else:
                self.frame_outcomes['collision'] += 1

            self.slot_counter += 1

            reader_bits = CONSTANTS.QUERY_CMD_BITS

            if num_responses == 0:
                return AlgorithmStepResult('idle_slot', reader_bits=reader_bits)

            if num_responses == 1:
                tag = responding_tags[0]
                result = AlgorithmStepResult('success_slot', reader_bits=reader_bits,
                                             tag_bits=CONSTANTS.EPC_CODE_BITS,
                                             expected_max_tag_bits=CONSTANTS.EPC_CODE_BITS)

                time_delta = calculate_time_delta(result)
                tag.identification_time_us = current_time_us + time_delta
                tag.is_identified = True
                self.identified_tags_count += 1
                return result
            else:

                total_tag_bits = num_responses * CONSTANTS.EPC_CODE_BITS
                return AlgorithmStepResult('collision_slot', reader_bits=reader_bits,
                                           tag_bits=total_tag_bits,
                                           expected_max_tag_bits=CONSTANTS.EPC_CODE_BITS)

        return None
