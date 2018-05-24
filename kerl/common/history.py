import datetime
import os.path
from typing import NamedTuple, List, Optional

import numpy as np


class HistoryRecord(NamedTuple):
    # Exact date and time of the record
    date_time: datetime.datetime
    # Exact total reward received during the simulation
    exact_reward: float
    # Moving average of the rewards
    average_reward: float
    # Number of observations since the beginning of training
    num_observations: int
    # Time in seconds passed since the last data point (last completed sim)
    diff_seconds: float
    # Number of observations since the last data point (last completed sim)
    diff_observations: int

    @classmethod
    def decode(cls, line: str) -> 'HistoryRecord':
        raw_items = line.split('\t')
        if len(raw_items) != len(cls._fields):
            raise ValueError('Invalid line')
        pieces = []
        # noinspection PyProtectedMember,PyUnresolvedReferences
        for (field, field_type), value in zip(cls._field_types.items(),
                                              raw_items):
            if field_type is datetime.datetime:
                decoded = datetime.datetime.strptime(
                    value, '%Y-%m-%dT%H:%M:%S.%f')
            else:
                decoded = field_type(value)
            pieces.append(decoded)
        return cls(*pieces)

    def encode(self) -> str:
        pieces = []
        # noinspection PyUnresolvedReferences
        for field, field_type in self._field_types.items():
            value = getattr(self, field)
            coded = (value.isoformat() if field_type is datetime.datetime
                     else str(value))
            pieces.append(coded)
        return '\t'.join(pieces)


class TrainHistoryRecorder:
    """
    Records moving average of all rewards received during the training
    along with the exact time and the number of observations seen by the agent.
    Later this allows to build graphs of learning curves, comparing
    speed, sample efficiency, etc.
    """
    def __init__(self, history_file_path: str,
                 num_steps: int,
                 average_reward_beta: float=0.9):
        self.history_file_path = history_file_path
        self.moving_average_reward = 0
        self.beta = average_reward_beta
        self.num_steps = num_steps
        self.record_buffer = []  # type: List[HistoryRecord]
        self.last_record = None  # type: Optional[HistoryRecord]
        if os.path.exists(history_file_path):
            with open(history_file_path, 'rt') as f:
                last_line = None
                for line in f:
                    if line:
                        last_line = line
                if last_line is not None:
                    try:
                        self.last_record = HistoryRecord.decode(last_line)
                    except ValueError:
                        pass
                    else:
                        self.moving_average_reward = (
                            self.last_record.average_reward)

    def record(self, step_idx: int,
               completed_episodes: np.ndarray,
               episode_rewards: np.ndarray):
        """
        Accumulates reward records in an internal buffer. It can be flushed
        on disk by calling `flush_records` method, which is normally done
        when the model is being saved. This helps to keep the records
        consistent if the training was interrupted.

        :param step_idx: current iteration of training
        :param completed_episodes: boolean mask of all simulations completed
            at the moment
        :param episode_rewards: an array of all rewards received
            in all simulations at the moment.
        """
        # if two simulations happened to complete simultaneously,
        # we just average their scores, but such event is highly
        # improbable
        mean_completed_reward = episode_rewards[completed_episodes].mean()
        if np.sum(completed_episodes) > 0:
            # Exponential moving average of the rewards
            self.moving_average_reward = (
                mean_completed_reward
                if self.moving_average_reward == 0
                else (self.beta * self.moving_average_reward +
                      (1 - self.beta) * mean_completed_reward))
            print('Average reward:', self.moving_average_reward,
                  'last exact reward:', mean_completed_reward)
            # recording
            total_observations = (
                    (step_idx + 1) *
                    self.num_steps * len(episode_rewards))
            if self.last_record is None:
                diff_seconds = 0
                diff_observations = total_observations
            else:
                diff_seconds = (
                   (datetime.datetime.now() - self.last_record.date_time)
                   .total_seconds())
                diff_observations = (
                    total_observations - self.last_record.num_observations)
            new_record = HistoryRecord(
                date_time=datetime.datetime.now(),
                exact_reward=mean_completed_reward,
                average_reward=self.moving_average_reward,
                num_observations=total_observations,
                diff_seconds=diff_seconds,
                diff_observations=diff_observations)
            self.record_buffer.append(new_record)
            self.last_record = new_record

    def flush_records(self):
        """
        Dumps recorded rewards and their timestamps on the disk.
        """
        with open(self.history_file_path, 'a+t') as h:
            for record in self.record_buffer:
                print(record.encode(), file=h)
        self.record_buffer.clear()


def read_history(file_path: str) -> np.ndarray:
    result = []
    with open(file_path, 'rt') as h:
        total_frames = 0
        for line in h:
            try:
                rec = HistoryRecord.decode(line)
            except ValueError:
                pass
            else:
                total_frames += rec.diff_observations
                result.append((rec.average_reward, total_frames))
    return np.array(result)
