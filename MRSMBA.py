from collections import deque
import math
from typing import Dict, List, Optional


from Framework import Tag, ReliableAlgorithmBase, AlgorithmStepResult, CONSTANTS


RN5_BITS = 5

TAGS_PER_COLLISION_SLOT = 2.26

OPTIMAL_TAG_DENSITY = 0.8


class MRSMBA_Algorithm(ReliableAlgorithmBase):
    def __init__(self, algorithm_specific_config: Dict):
        super().__init__(algorithm_specific_config)

        self.current_procedure = 'IDLE'

        self.reservation_slots_total = 0
        self.reservation_slot_counter = 0
        self.reservation_outcomes = {'success': 0, 'collision': 0}

        self.identification_queue: deque[List[Tag]] = deque()

        self.history_collisions = deque([0, 0], maxlen=2)
        self.history_frame_len = deque([1, 1], maxlen=2)
        self.estimated_arrival_rate = 0.0

    def get_backlog_size(self) -> int:
        return sum(len(tags) for tags in self.identification_queue)

    def _estimate_next_round_params(self):

        l_im1_c = self.history_collisions[1]
        l_im2_c = self.history_collisions[0]
        l_im1 = self.history_frame_len[1]

        if l_im1 > 0:
            numerator = l_im1_c + TAGS_PER_COLLISION_SLOT * \
                l_im1_c - TAGS_PER_COLLISION_SLOT * l_im2_c
            self.estimated_arrival_rate = numerator / l_im1

        l_ic = self.reservation_outcomes['collision']

        denominator = OPTIMAL_TAG_DENSITY - self.estimated_arrival_rate
        if denominator <= 0:

            optimal_slots = 512
        else:
            optimal_slots = (TAGS_PER_COLLISION_SLOT * l_ic) / denominator

        if optimal_slots <= 0:
            optimal_slots = 2

        q_value = math.ceil(math.log2(optimal_slots))
        self.reservation_slots_total = int(2**q_value)

        self.history_collisions.append(self.reservation_outcomes['collision'])
        self.history_frame_len.append(self.reservation_outcomes['success'])
        self.reservation_outcomes = {'success': 0, 'collision': 0}

    def execute_step(self, active_tags: List[Tag], current_time_us: float) -> Optional[AlgorithmStepResult]:

        if self.current_procedure == 'IDLE':

            self._estimate_next_round_params()
            self.reservation_slot_counter = 0
            self.current_procedure = 'RESERVATION'

        if self.current_procedure == 'RESERVATION':
            if self.reservation_slot_counter < self.reservation_slots_total:

                slot_idx = self.reservation_slot_counter

                responding_tags = [
                    tag for tag in active_tags
                    if not tag.is_identified and (int(tag.id, 2) % self.reservation_slots_total == slot_idx)
                ]

                responding_tags = self.get_actual_responders(responding_tags)
                num_responses = len(responding_tags)

                self.reservation_slot_counter += 1

                reader_bits = CONSTANTS.QUERY_CMD_BITS

                if num_responses == 0:

                    return AlgorithmStepResult(
                        operation_type='idle_slot',
                        reader_bits=reader_bits,
                        operation_description=f"MRSMBA-Res: Idle in slot {slot_idx}"
                    )
                elif num_responses == 1:

                    self.reservation_outcomes['success'] += 1
                    self.identification_queue.append(responding_tags)

                    return AlgorithmStepResult(
                        operation_type='success_slot',
                        reader_bits=reader_bits,
                        tag_bits=RN5_BITS,
                        expected_max_tag_bits=RN5_BITS,
                        operation_description=f"MRSMBA-Res: Success in slot {slot_idx}"
                    )
                else:

                    self.reservation_outcomes['collision'] += 1

                    return AlgorithmStepResult(
                        operation_type='collision_slot',
                        reader_bits=reader_bits,
                        tag_bits=num_responses * RN5_BITS,
                        expected_max_tag_bits=RN5_BITS,
                        operation_description=f"MRSMBA-Res: Collision in slot {slot_idx}"
                    )
            else:

                self.current_procedure = 'IDENTIFICATION'

        if self.current_procedure == 'IDENTIFICATION':
            if self.identification_queue:

                tags_to_identify = self.identification_queue.popleft()

                tags_to_identify = [
                    t for t in tags_to_identify if t in active_tags]

                tags_to_identify = self.get_actual_responders(tags_to_identify)

                if not tags_to_identify:
                    return AlgorithmStepResult('internal_op', operation_description="MRSMBA-ID: Tag left before identification.")

                num_to_identify = len(tags_to_identify)
                reader_bits = CONSTANTS.QUERY_CMD_BITS

                if num_to_identify == 1:
                    tag = tags_to_identify[0]
                    tag.is_identified = True
                    tag.identification_time_us = current_time_us
                    self.identified_tags_count += 1

                    return AlgorithmStepResult(
                        operation_type='success_slot',
                        reader_bits=reader_bits,
                        tag_bits=CONSTANTS.EPC_CODE_BITS,
                        expected_max_tag_bits=CONSTANTS.EPC_CODE_BITS,
                        operation_description=f"MRSMBA-ID: Success for tag {tag.id[-6:]}"
                    )
                else:

                    return AlgorithmStepResult(
                        operation_type='collision_slot',
                        reader_bits=reader_bits,
                        tag_bits=num_to_identify * CONSTANTS.EPC_CODE_BITS,
                        expected_max_tag_bits=CONSTANTS.EPC_CODE_BITS,
                        operation_description=f"MRSMBA-ID: Collision with {num_to_identify} tags."
                    )
            else:

                self.current_procedure = 'IDLE'
                return AlgorithmStepResult('internal_op', operation_description="MRSMBA: Round finished.")

        return None
