

"""
Framework-MRS (Mobile RFID Simulation)
"""

import random
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any


@dataclass(frozen=True)
class SimulationConstants:
    TARI_US: float = 12.5
    @property
    def RTCAL_US(self) -> float: return 2.5 * self.TARI_US
    @property
    def READER_TO_TAG_BPS(self) -> float: return 1.0 / (self.TARI_US * 1e-6)
    @property
    def TAG_TO_READER_BPS(self) -> float: return 2.0 * self.READER_TO_TAG_BPS
    @property
    def T1_US(self) -> float: return max(self.RTCAL_US, 10.0)
    @property
    def T2_MIN_US(self) -> float: return self.T1_US + 20.0

    @property
    def T4_US(self) -> float: return 2.0 * self.TARI_US

    QUERY_CMD_BITS: int = 22
    ACK_CMD_BITS: int = 18
    RN16_RESPONSE_BITS: int = 16
    EPC_CODE_BITS: int = 96

    @property
    def READER_BITS_PER_US(
        self) -> float: return self.READER_TO_TAG_BPS / 1.0e6

    @property
    def TAG_BITS_PER_US(self) -> float: return self.TAG_TO_READER_BPS / 1.0e6


CONSTANTS = SimulationConstants()


class AlgorithmStepResult:
    def __init__(self, operation_type: str, reader_bits: float = 0.0, tag_bits: float = 0.0,
                 expected_max_tag_bits: int = 0, operation_description: str = '',
                 override_time_us: Optional[float] = None):
        self.operation_type = operation_type
        self.reader_bits = reader_bits
        self.tag_bits = tag_bits
        self.expected_max_tag_bits = expected_max_tag_bits
        self.operation_description = operation_description
        self.override_time_us = override_time_us


@dataclass
class StreamScenarioConfig:
    simulation_duration_s: float = 10.0
    tag_arrival_rate_per_s: int = 100
    mean_residence_time_ms: float = 2000.0
    std_dev_residence_time_ms: float = 500.0
    tag_id_length: int = 96

    tag_failure_probability: float = 0.0


class Tag:
    def __init__(self, tag_id: str, entry_time_us: float, exit_time_us: float):
        self.id: str = tag_id
        self.entry_time_us: float = entry_time_us
        self.exit_time_us: float = exit_time_us
        self.status: str = 'FRESH'
        self.context_id: Optional[Any] = None
        self.is_identified: bool = False
        self.identification_time_us: Optional[float] = None


class MobileAlgorithmInterface:
    def __init__(self, algorithm_specific_config: Dict):
        self.config = algorithm_specific_config
        self.identified_tags_count = 0

    def execute_step(self, active_tags: List[Tag], current_time_us: float) -> Optional[AlgorithmStepResult]:
        raise NotImplementedError

    def get_backlog_size(self) -> int:
        return 0


class ReliableAlgorithmBase(MobileAlgorithmInterface):
    def __init__(self, algorithm_specific_config: dict):
        super().__init__(algorithm_specific_config)
        self.failure_prob = algorithm_specific_config.get(
            'tag_failure_probability', 0.0)

    def get_actual_responders(self, intended_responders: List[Tag]) -> List[Tag]:

        if self.failure_prob == 0.0 or not intended_responders:
            return intended_responders

        return [tag for tag in intended_responders if random.random() > self.failure_prob]


def generate_tag_stream(config: StreamScenarioConfig) -> List[Tag]:
    tags = []
    current_time_us = 0.0
    if config.tag_arrival_rate_per_s > 0:
        total_tags_to_generate = int(
            config.simulation_duration_s * config.tag_arrival_rate_per_s * 1.5) + 200
        for _ in range(total_tags_to_generate):
            time_to_next_arrival_s = random.expovariate(
                config.tag_arrival_rate_per_s)
            current_time_us += time_to_next_arrival_s * 1e6
            entry_time = current_time_us
            residence_time_ms = random.normalvariate(
                config.mean_residence_time_ms, config.std_dev_residence_time_ms)
            residence_time_us = max(1, residence_time_ms) * 1e3
            exit_time = entry_time + residence_time_us
            tag_id = ''.join(random.choice('01')
                             for _ in range(config.tag_id_length))
            tags.append(Tag(tag_id, entry_time, exit_time))
    return tags


def calculate_time_delta(step_result: AlgorithmStepResult) -> float:
    if step_result.override_time_us is not None:
        return step_result.override_time_us
    if step_result.operation_type == 'internal_op':
        return 0.0

    time_reader_tx = step_result.reader_bits / CONSTANTS.READER_BITS_PER_US

    if step_result.operation_type == 'idle_slot':
        return time_reader_tx + CONSTANTS.T4_US

    elif step_result.operation_type in ['success_slot', 'collision_slot']:
        time_tag_response = step_result.expected_max_tag_bits / CONSTANTS.TAG_BITS_PER_US
        return time_reader_tx + CONSTANTS.T1_US + time_tag_response + CONSTANTS.T2_MIN_US

    return 0.0


def run_simulation(
    scenario_config: StreamScenarioConfig,
    algorithm_class,
    algorithm_specific_config: Dict,
    time_series_interval_ms: Optional[float] = None,
    pregenerated_tags: Optional[List[Tag]] = None
) -> Dict:

    if pregenerated_tags is not None:
        all_tags = pregenerated_tags
    else:
        all_tags = generate_tag_stream(scenario_config)

    if hasattr(scenario_config, 'tag_failure_probability'):
        algorithm_specific_config['tag_failure_probability'] = scenario_config.tag_failure_probability

    algo_instance = algorithm_class(algorithm_specific_config)

    current_time_us = 0.0
    simulation_duration_us = scenario_config.simulation_duration_s * 1e6
    total_busy_time_us = 0.0
    raw_counters = {'total_reader_bits': 0.0,
                    'total_tag_bits': 0.0, 'total_steps': 0}

    events = []
    for tag in all_tags:
        if tag.entry_time_us < simulation_duration_us:
            events.append((tag.entry_time_us, 'entry', tag))
        if tag.exit_time_us < simulation_duration_us:
            events.append((tag.exit_time_us, 'exit', tag))
    events.sort(key=lambda x: x[0])

    present_tags_set = set()
    event_idx = 0

    time_series_log = []
    log_interval_us = None
    next_log_time_us = float('inf')

    interval_stats = {
        'unidentified_count_samples': [],
        'age_of_oldest_unidentified_samples': [],
        'internal_backlog_size_samples': [],
        'channel_efficiency_samples': []
    }
    interval_counters = {
        'identified_count': 0,
        'lost_count': 0,
        'busy_time_us': 0.0,
        'success_time_us': 0.0
    }

    if time_series_interval_ms is not None and time_series_interval_ms > 0:
        log_interval_us = time_series_interval_ms * 1000.0
        next_log_time_us = log_interval_us

    while current_time_us < simulation_duration_us:

        if current_time_us >= next_log_time_us:

            interval_duration_s = log_interval_us / 1e6
            log_entry = {'time_ms': next_log_time_us / 1e3}
            log_entry['interval_throughput_tps'] = interval_counters['identified_count'] / \
                interval_duration_s
            log_entry['interval_loss_rate_tps'] = interval_counters['lost_count'] / \
                interval_duration_s

            for key, samples in interval_stats.items():
                metric_name = key.replace('_samples', '')
                if samples:
                    log_entry[f'{metric_name}_mean'] = np.mean(samples)
                    log_entry[f'{metric_name}_max'] = np.max(samples)
                    log_entry[f'{metric_name}_min'] = np.min(samples)
                    log_entry[f'{metric_name}_std'] = np.std(samples)
                else:
                    log_entry[f'{metric_name}_mean'] = 0.0
                    log_entry[f'{metric_name}_max'] = 0.0
                    log_entry[f'{metric_name}_min'] = 0.0
                    log_entry[f'{metric_name}_std'] = 0.0

            time_series_log.append(log_entry)

            for key in interval_stats:
                interval_stats[key].clear()
            for key in interval_counters:
                interval_counters[key] = 0.0 if isinstance(
                    interval_counters[key], float) else 0

            next_log_time_us += log_interval_us

        last_present_count = len(present_tags_set)

        while event_idx < len(events) and events[event_idx][0] <= current_time_us:
            _event_time, event_type, tag = events[event_idx]
            if event_type == 'entry':
                present_tags_set.add(tag)
            elif event_type == 'exit':
                present_tags_set.discard(tag)
                if not tag.is_identified:
                    interval_counters['lost_count'] += 1
            event_idx += 1

        last_identified_count = algo_instance.identified_tags_count
        step_result = algo_instance.execute_step(
            list(present_tags_set), current_time_us)

        if step_result:
            time_delta = calculate_time_delta(step_result)

            identified_this_step = algo_instance.identified_tags_count - last_identified_count
            if identified_this_step > 0:
                interval_counters['identified_count'] += identified_this_step

            interval_counters['busy_time_us'] += time_delta
            if step_result.operation_type == 'success_slot':
                interval_counters['success_time_us'] += time_delta

            total_busy_time_us += time_delta
            raw_counters['total_reader_bits'] += step_result.reader_bits
            raw_counters['total_tag_bits'] += step_result.tag_bits
            raw_counters['total_steps'] += 1

            next_step_time_us = current_time_us + time_delta
        else:
            if event_idx < len(events):
                next_step_time_us = events[event_idx][0]
            else:
                break

        if log_interval_us is not None:
            unidentified_tags_in_scene = [
                t for t in present_tags_set if not t.is_identified]

            interval_stats['unidentified_count_samples'].append(
                len(unidentified_tags_in_scene))

            if unidentified_tags_in_scene:
                oldest_entry_time = min(
                    t.entry_time_us for t in unidentified_tags_in_scene)
                age_ms = (current_time_us - oldest_entry_time) / 1e3
                interval_stats['age_of_oldest_unidentified_samples'].append(
                    age_ms)
            else:
                interval_stats['age_of_oldest_unidentified_samples'].append(
                    0.0)

            interval_stats['internal_backlog_size_samples'].append(
                algo_instance.get_backlog_size())

            efficiency = interval_counters['success_time_us'] / \
                interval_counters['busy_time_us'] if interval_counters['busy_time_us'] > 0 else 0.0
            interval_stats['channel_efficiency_samples'].append(efficiency)

        current_time_us = next_step_time_us

    raw_results = {
        "all_tags": all_tags,
        "total_busy_time_us": total_busy_time_us,
        "simulation_end_time_us": current_time_us,
        "raw_counters": raw_counters
    }
    if time_series_log:
        raw_results['time_series_data'] = time_series_log

    return raw_results
