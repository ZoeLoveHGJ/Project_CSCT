
from collections import deque
import math
from typing import Dict, List, Optional, Set


from Framework import Tag, ReliableAlgorithmBase, AlgorithmStepResult, CONSTANTS, calculate_time_delta
from Tool import RfidUtils

def dec2bin(number: int, bits: int) -> str:
    if bits == 0:
        return ''
    return format(number, f'0{bits}b')

class ACDQT_Algorithm(ReliableAlgorithmBase):
    def __init__(self, algorithm_specific_config: Dict):
        super().__init__(algorithm_specific_config)
        
        
        self.phase = 'INIT'  

        
        self.round_count = 0
        self.round_start_time_us = 0.0
        self.estimation_data = {'T': [], 'n': []}
        self.tag_flow_rate = 0.0
        self.avg_id_time_us = 0.0
        self.wait_until_time_us = 0.0
        self.admitted_tags_heuristic = algorithm_specific_config.get('admitted_tags_heuristic', 100)
        self.target_loss_ratio = algorithm_specific_config.get('target_loss_ratio', 0.01)

        
        self.query_queue: deque[tuple[str, int]] = deque()
        self.tags_in_current_round: Set[Tag] = set()
        
        
        self.frame_in_progress = False
        self.frame_prefix = ''
        self.frame_k = 0
        self.frame_slot_idx = 0
        self.frame_tags_by_slot: Dict[int, List[Tag]] = {} 
        self.frame_new_tasks = []

    def get_backlog_size(self) -> int:
        return len(self.query_queue)

    def _start_new_round(self, active_tags: List[Tag], current_time_us: float):
        self.round_count += 1
        self.round_start_time_us = current_time_us
        self.tags_in_current_round = {t for t in active_tags if not t.is_identified}
        
        self.query_queue.clear()
        self.query_queue.append(('', 0))
        self.frame_in_progress = False
        self.phase = 'RECOGNITION'

    def _finish_round(self, current_time_us: float):
        round_duration_us = current_time_us - self.round_start_time_us
        identified_in_round = len([t for t in self.tags_in_current_round if t.is_identified])

        if 1 <= self.round_count <= 3:
            self.estimation_data['T'].append(round_duration_us)
            self.estimation_data['n'].append(identified_in_round)
        
        if self.round_count == 3:
            total_duration_s = sum(self.estimation_data['T']) / 1e6
            total_tags = sum(self.estimation_data['n'])
            if total_duration_s > 0:
                self.tag_flow_rate = total_tags / total_duration_s
            if total_tags > 0:
                self.avg_id_time_us = sum(self.estimation_data['T']) / total_tags
        
        if self.round_count >= 3:
            if self.tag_flow_rate > 0:
                wait_time_s = self.admitted_tags_heuristic / self.tag_flow_rate
                self.wait_until_time_us = current_time_us + (wait_time_s * 1e6)
                self.phase = 'TIME_WAIT'
            else:
                self.phase = 'IDLE_WAIT'
        else:
            self.phase = 'ESTIMATION'

    def execute_step(self, active_tags: List[Tag], current_time_us: float) -> Optional[AlgorithmStepResult]:
        
        if self.phase == 'INIT':
            self.phase = 'ESTIMATION'

        if self.phase == 'ESTIMATION':
            self._start_new_round(active_tags, current_time_us)
        
        if self.phase == 'TIME_WAIT':
            if current_time_us >= self.wait_until_time_us:
                self._start_new_round(active_tags, current_time_us)
            else:
                return None
        
        if self.phase == 'IDLE_WAIT':
            
            unidentified_tags_present = any(not t.is_identified for t in active_tags)
            if unidentified_tags_present:
                
                self.phase = 'ESTIMATION'
                self.round_count = 0
                self.estimation_data = {'T': [], 'n': []}
            

        
        if self.phase == 'RECOGNITION':
            still_active_targets = self.tags_in_current_round.intersection(active_tags)
            has_work_to_do = any(not tag.is_identified for tag in still_active_targets)

            if not has_work_to_do and not self.query_queue and not self.frame_in_progress:
                self._finish_round(current_time_us)
                return AlgorithmStepResult('internal_op', operation_description=f"ACDQT: Round {self.round_count} aborted (no active targets left).")

            if not self.frame_in_progress:
                if not self.query_queue:
                    self._finish_round(current_time_us)
                    return AlgorithmStepResult('internal_op', operation_description=f"ACDQT: Round {self.round_count} finished.")

                
                self.frame_prefix, self.frame_k = self.query_queue.popleft()
                self.frame_in_progress = True
                self.frame_slot_idx = 0
                self.frame_new_tasks = []
                
                
                self.frame_tags_by_slot.clear()
                participating_tags = [
                    t for t in self.tags_in_current_round 
                    if not t.is_identified and t.id.startswith(self.frame_prefix)
                ]
                
                if self.frame_k > 0:
                    for tag in participating_tags:
                        prefix_len = len(self.frame_prefix)
                        if len(tag.id) >= prefix_len + self.frame_k:
                            slot_index_str = tag.id[prefix_len : prefix_len + self.frame_k]
                            if slot_index_str:
                                slot_index = int(slot_index_str, 2)
                                if slot_index not in self.frame_tags_by_slot:
                                    self.frame_tags_by_slot[slot_index] = []
                                self.frame_tags_by_slot[slot_index].append(tag)
                else: 
                    self.frame_tags_by_slot[0] = participating_tags

            
            current_slot = self.frame_slot_idx
            responding_tags = self.frame_tags_by_slot.get(current_slot, [])
            num_responses = len(responding_tags)
            
            
            responding_tags = self.get_actual_responders(responding_tags)
            num_responses = len(responding_tags)
            
            reader_bits = CONSTANTS.QUERY_CMD_BITS + len(self.frame_prefix) + 2 + 16

            result = None
            if num_responses == 0:
                result = AlgorithmStepResult('idle_slot', reader_bits=reader_bits)
            else:
                remaining_bits_len = CONSTANTS.EPC_CODE_BITS - len(self.frame_prefix) - self.frame_k
                if remaining_bits_len < 0: remaining_bits_len = 0

                if num_responses == 1:
                    tag = responding_tags[0]
                    result = AlgorithmStepResult('success_slot', reader_bits=reader_bits,
                                                 tag_bits=remaining_bits_len, expected_max_tag_bits=remaining_bits_len)
                    time_delta = calculate_time_delta(result)
                    tag.identification_time_us = current_time_us + time_delta
                    tag.is_identified = True
                    self.identified_tags_count += 1
                else: 
                    total_tag_bits_for_cost = num_responses * remaining_bits_len
                    
                    
                    
                    slot_binary = dec2bin(current_slot, self.frame_k)
                    new_base_prefix = self.frame_prefix + slot_binary

                    
                    remaining_ids = [t.id[len(new_base_prefix):] for t in responding_tags]
                    lcp = RfidUtils.get_collision_info(remaining_ids)[0]
                    
                    final_new_prefix = new_base_prefix + lcp
                    
                    
                    new_k = 1 
                    
                    self.frame_new_tasks.append((final_new_prefix, new_k))
                    
                    
                    result = AlgorithmStepResult('collision_slot', reader_bits=reader_bits,
                                                 tag_bits=total_tag_bits_for_cost, expected_max_tag_bits=remaining_bits_len)
            
            
            self.frame_slot_idx += 1
            if self.frame_slot_idx >= (2**self.frame_k):
                
                self.frame_in_progress = False
                for task in self.frame_new_tasks:
                    self.query_queue.append(task)

            return result

        return None
