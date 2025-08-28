import math
from collections import deque
from typing import Dict, List, Optional, Set


from Framework import Tag, MobileAlgorithmInterface, AlgorithmStepResult, CONSTANTS, calculate_time_delta
from Tool import RfidUtils


class EDRCT_Algorithm(MobileAlgorithmInterface):
    def __init__(self, algorithm_specific_config: Dict):
        super().__init__(algorithm_specific_config)

        self.task_stack: deque[tuple[str, int]] = deque()

        self.tag_search_depths: Dict[str, int] = {}

        self.tags_in_cycle: Set[Tag] = set()

        self.last_reset_time_us: float = 0.0

        initial_interval = algorithm_specific_config.get(
            'initial_arrival_interval_us', 150 * 1000)
        self.estimated_arrival_interval_us: float = initial_interval

        self.alpha: float = algorithm_specific_config.get('alpha', 0.8)

        self.batch_start_time_us: float = 0.0

    def get_backlog_size(self) -> int:
        return len(self.task_stack)

    def _get_tag_depth(self, tag: Tag) -> int:
        return self.tag_search_depths.get(tag.id, 0)

    def _start_new_cycle(self, active_tags: List[Tag], current_time_us: float, is_global_reset: bool):
        if is_global_reset:
            self.tag_search_depths.clear()

        self.task_stack.clear()

        if is_global_reset:
            self.last_reset_time_us = current_time_us

        self.batch_start_time_us = current_time_us

        self.tags_in_cycle = {t for t in active_tags if not t.is_identified}

        if self.tags_in_cycle:

            self.task_stack.append(('', 0))

    def execute_step(self, active_tags: List[Tag], current_time_us: float) -> Optional[AlgorithmStepResult]:

        time_since_last_reset = current_time_us - self.last_reset_time_us
        if time_since_last_reset > self.estimated_arrival_interval_us:
            self._start_new_cycle(
                active_tags, current_time_us, is_global_reset=True)
            return AlgorithmStepResult('internal_op', operation_description="EDRCT: Time-triggered global reset.")

        if not self.task_stack:

            unidentified_tags_in_field = {
                t for t in active_tags if not t.is_identified}
            if unidentified_tags_in_field:
                instantaneous_interval_us = current_time_us - self.batch_start_time_us
                if instantaneous_interval_us > 0:
                    self.estimated_arrival_interval_us = (self.alpha * instantaneous_interval_us) + \
                                                         ((1 - self.alpha) *
                                                          self.estimated_arrival_interval_us)

                self._start_new_cycle(
                    active_tags, current_time_us, is_global_reset=False)
            else:

                return None

        if not self.task_stack:
            return None

        query_prefix, query_depth = self.task_stack.pop()

        responding_tags = [
            tag for tag in self.tags_in_cycle
            if not tag.is_identified
            and self._get_tag_depth(tag) == query_depth
            and tag.id.startswith(query_prefix)
        ]
        num_responses = len(responding_tags)

        reader_bits = CONSTANTS.QUERY_CMD_BITS + len(query_prefix)

        if num_responses == 0:
            return AlgorithmStepResult('idle_slot', reader_bits=reader_bits)

        remaining_bits_len = CONSTANTS.EPC_CODE_BITS - len(query_prefix)

        if num_responses == 1:
            tag = responding_tags[0]
            result = AlgorithmStepResult('success_slot', reader_bits=reader_bits,
                                         tag_bits=remaining_bits_len, expected_max_tag_bits=remaining_bits_len)

            time_delta = calculate_time_delta(result)
            tag.identification_time_us = current_time_us + time_delta
            tag.is_identified = True
            self.identified_tags_count += 1
            return result
        else:

            total_tag_bits = num_responses * remaining_bits_len

            lcp = RfidUtils.get_collision_info(
                [t.id for t in responding_tags])[0]

            self.task_stack.append((lcp, query_depth))

            self.task_stack.append((lcp, query_depth + 1))

            for tag in responding_tags:
                self.tag_search_depths[tag.id] = query_depth + 1

            return AlgorithmStepResult('collision_slot', reader_bits=reader_bits,
                                       tag_bits=total_tag_bits, expected_max_tag_bits=remaining_bits_len)
