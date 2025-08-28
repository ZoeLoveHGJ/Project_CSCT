import os
from collections import deque
from typing import Dict, List, Set, Optional, Any


from Framework import Tag, ReliableAlgorithmBase, AlgorithmStepResult, CONSTANTS


class DCT_Algorithm(ReliableAlgorithmBase):
    def __init__(self, algorithm_specific_config: Dict):
        super().__init__(algorithm_specific_config)

        self.prefix_stack: deque[str] = deque()

    def get_backlog_size(self) -> int:

        return len(self.prefix_stack)

    def execute_step(self, active_tags: List[Tag], current_time_us: float) -> Optional[AlgorithmStepResult]:

        if not self.prefix_stack:
            self.prefix_stack.append('')

        query_prefix = self.prefix_stack.pop()

        responding_tags = [
            tag for tag in active_tags
            if not tag.is_identified and tag.id.startswith(query_prefix)
        ]

        responding_tags = self.get_actual_responders(responding_tags)
        num_responses = len(responding_tags)

        reader_bits = CONSTANTS.QUERY_CMD_BITS + len(query_prefix)

        if num_responses == 0:

            return AlgorithmStepResult(
                operation_type='idle_slot',
                reader_bits=reader_bits,
                operation_description=f"DCT: Idle on prefix '{query_prefix}'"
            )

        elif num_responses == 1:

            tag_to_identify = responding_tags[0]
            tag_to_identify.is_identified = True
            tag_to_identify.identification_time_us = current_time_us
            self.identified_tags_count += 1

            tag_reply_bits = CONSTANTS.EPC_CODE_BITS - len(query_prefix)

            return AlgorithmStepResult(
                operation_type='success_slot',
                reader_bits=reader_bits,
                tag_bits=tag_reply_bits,
                expected_max_tag_bits=tag_reply_bits,
                operation_description=f"DCT: Success for tag {tag_to_identify.id[-6:]}"
            )

        else:

            colliding_ids_suffix = [
                t.id[len(query_prefix):] for t in responding_tags]
            lcp = os.path.commonprefix(colliding_ids_suffix)

            prefix1 = query_prefix + lcp + '1'
            prefix0 = query_prefix + lcp + '0'
            self.prefix_stack.append(prefix1)
            self.prefix_stack.append(prefix0)

            remaining_bits_per_tag = CONSTANTS.EPC_CODE_BITS - \
                len(query_prefix)

            total_tag_bits_for_cost = num_responses * remaining_bits_per_tag

            expected_bits_for_duration = remaining_bits_per_tag

            return AlgorithmStepResult(
                operation_type='collision_slot',
                reader_bits=reader_bits,
                tag_bits=total_tag_bits_for_cost,
                expected_max_tag_bits=expected_bits_for_duration,
                operation_description=f"DCT: Collision on '{query_prefix}'. Split into '{prefix0}' and '{prefix1}'"
            )
