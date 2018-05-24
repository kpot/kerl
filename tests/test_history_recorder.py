import datetime

from kerl.common.history import HistoryRecord


def test_history_record_encoding():
    orig_record = HistoryRecord(
        date_time=datetime.datetime.now(),
        exact_reward=2,
        average_reward=10,
        num_observations=1000,
        diff_seconds=5,
        diff_observations=15)
    encoded_record = orig_record.encode()
    decoded_record = HistoryRecord.decode(encoded_record)
    assert decoded_record == orig_record

