import math
import heapq
from typing import Dict, List, Optional, Any, Tuple


from Framework import MobileAlgorithmInterface, Tag, AlgorithmStepResult, CONSTANTS, calculate_time_delta
from Tool import RfidUtils


class CollisionContext:
    def __init__(self, context_id: int, tags: List[Tag], estimated_size: float, prefix: str = ''):
        self.id: int = context_id
        self.tags: List[Tag] = tags
        self.prefix: str = prefix
        self.estimated_size: float = estimated_size

    def __lt__(self, other):
        return self.id < other.id


class CSCT_Est(MobileAlgorithmInterface):
    def __init__(self, algorithm_specific_config: Dict):
        super().__init__(algorithm_specific_config)
        self.ccq: list = []
        self.next_context_id: int = 0
        self.q_value: float = algorithm_specific_config.get('initial_q', 11.0)
        self.frame_size: int = int(2**self.q_value)
        self.slot_counter: int = 0
        self.slot_outcomes = {'idle': 0, 'success': 0, 'collision': 0}
        self.current_phase: str = 'A'
        self._transient_collision_tags: Dict[int, List[Tag]] = {}

        self.priority_weight_size: float = algorithm_specific_config.get(
            'priority_weight_size', 1.0)

        self.priority_weight_depth: float = algorithm_specific_config.get(
            'priority_weight_depth', 0.001)

        self.estimator_multiplier: float = 2.39

    def get_backlog_size(self) -> int:
        return len(self.ccq)

    def execute_step(self, active_tags: List[Tag], current_time_us: float) -> Optional[AlgorithmStepResult]:

        if self.current_phase == 'A':
            if self.slot_counter < self.frame_size:
                return self._phase_A_discover_slot(active_tags, current_time_us)
            else:
                self._process_frame_results()
                self.current_phase = 'B'
                return self._phase_B_resolve(active_tags, current_time_us)
        elif self.current_phase == 'B':
            result = self._phase_B_resolve(active_tags, current_time_us)
            if result is None:
                self.current_phase = 'A'
                return self._phase_A_discover_slot(active_tags, current_time_us)
            return result
        return None

    def _calculate_priority(self, estimated_size: float, prefix_len: int) -> float:
        return (self.priority_weight_size * estimated_size) - (self.priority_weight_depth * prefix_len)

    def _phase_A_discover_slot(self, active_tags: List[Tag], current_time_us: float) -> AlgorithmStepResult:

        fresh_tags = [tag for tag in active_tags if tag.status == 'FRESH']
        safe_frame_size = max(1, self.frame_size)
        responding_tags = [tag for tag in fresh_tags if int(
            tag.id, 2) % safe_frame_size == self.slot_counter]

        num_responders = len(responding_tags)
        reader_bits = CONSTANTS.QUERY_CMD_BITS
        result: AlgorithmStepResult
        if num_responders == 0:
            self.slot_outcomes['idle'] += 1
            result = AlgorithmStepResult('idle_slot', reader_bits=reader_bits)
        elif num_responders == 1:
            self.slot_outcomes['success'] += 1
            tag = responding_tags[0]
            result = AlgorithmStepResult('success_slot', reader_bits=reader_bits,
                                         tag_bits=CONSTANTS.RN16_RESPONSE_BITS, expected_max_tag_bits=CONSTANTS.RN16_RESPONSE_BITS)
            time_delta = calculate_time_delta(result)
            tag.identification_time_us = current_time_us + time_delta
            tag.is_identified = True
            tag.status = 'IDENTIFIED'
            self.identified_tags_count += 1
        else:
            self.slot_outcomes['collision'] += 1
            self._transient_collision_tags[self.slot_counter] = responding_tags
            result = AlgorithmStepResult('collision_slot', reader_bits=reader_bits, tag_bits=num_responders *
                                         CONSTANTS.RN16_RESPONSE_BITS, expected_max_tag_bits=CONSTANTS.RN16_RESPONSE_BITS)
            for tag in responding_tags:
                tag.status = 'COLLIDED'
        self.slot_counter += 1
        return result

    def _process_frame_results(self):

        num_collisions = self.slot_outcomes['collision']

        if num_collisions > 0:

            estimated_size_per_context = self.estimator_multiplier
        else:

            estimated_size_per_context = 2.0

        for slot_index, tags in self._transient_collision_tags.items():
            new_context = CollisionContext(
                context_id=self.next_context_id,
                tags=tags,
                estimated_size=estimated_size_per_context,
                prefix=''
            )
            self.next_context_id += 1
            priority = self._calculate_priority(new_context.estimated_size, 0)
            heapq.heappush(self.ccq, (priority, new_context))

        C = 0.2
        idle = self.slot_outcomes['idle']
        collision = self.slot_outcomes['collision']
        if collision > idle:
            self.q_value = min(15.0, self.q_value + C)
        elif collision < idle:
            self.q_value = max(0.0, self.q_value - C)
        self.frame_size = int(round(2**self.q_value))

        self.slot_counter = 0
        self.slot_outcomes = {'idle': 0, 'success': 0, 'collision': 0}
        self._transient_collision_tags.clear()

    def _phase_B_resolve(self, active_tags: List[Tag], current_time_us: float) -> Optional[AlgorithmStepResult]:

        if not self.ccq:
            return None
        _priority, context = heapq.heappop(self.ccq)
        context.tags = [
            tag for tag in context.tags if tag in active_tags and not tag.is_identified]
        if not context.tags:
            return AlgorithmStepResult('internal_op')
        if len(context.tags) == 1:
            tag = context.tags[0]
            reader_request_bits = CONSTANTS.QUERY_CMD_BITS + \
                len(context.prefix)
            remaining_bits = CONSTANTS.EPC_CODE_BITS - len(context.prefix)
            result = AlgorithmStepResult('success_slot', reader_bits=reader_request_bits,
                                         tag_bits=remaining_bits, expected_max_tag_bits=remaining_bits)
            time_delta = calculate_time_delta(result)
            tag.identification_time_us = current_time_us + time_delta
            tag.is_identified = True
            tag.status = 'IDENTIFIED'
            self.identified_tags_count += 1
            return result
        else:
            return self._split_collision(context)

    def _split_collision(self, context: CollisionContext) -> AlgorithmStepResult:

        tag_ids = [t.id for t in context.tags]
        lcp, collision_positions = RfidUtils.get_collision_info(tag_ids)
        if not collision_positions:
            return AlgorithmStepResult('internal_op')

        split_pos = collision_positions[0]
        reader_request_bits = CONSTANTS.QUERY_CMD_BITS + len(context.prefix)
        remaining_bits_per_tag = CONSTANTS.EPC_CODE_BITS - len(context.prefix)
        result = AlgorithmStepResult('collision_slot', reader_bits=reader_request_bits, tag_bits=len(
            context.tags) * remaining_bits_per_tag, expected_max_tag_bits=remaining_bits_per_tag)

        group_0 = [t for t in context.tags if t.id[split_pos] == '0']
        group_1 = [t for t in context.tags if t.id[split_pos] == '1']

        est_size_for_children = max(1.0, context.estimated_size / 2.0)

        if group_1:
            new_prefix_1 = lcp + '1'
            new_context_1 = CollisionContext(
                self.next_context_id, group_1, est_size_for_children, new_prefix_1)
            self.next_context_id += 1
            priority = self._calculate_priority(
                new_context_1.estimated_size, len(new_prefix_1))
            heapq.heappush(self.ccq, (priority, new_context_1))
        if group_0:
            new_prefix_0 = lcp + '0'
            new_context_0 = CollisionContext(
                self.next_context_id, group_0, est_size_for_children, new_prefix_0)
            self.next_context_id += 1
            priority = self._calculate_priority(
                new_context_0.estimated_size, len(new_prefix_0))
            heapq.heappush(self.ccq, (priority, new_context_0))

        return result
