from concurrent.futures import ThreadPoolExecutor

from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id


def _store(path):
    return IncrementalJsonlStore(path, lock_timeout_seconds=2.0, lock_poll_seconds=0.005)


def test_trial_ids_are_deterministic_order_independent_and_unique():
    first = stable_trial_id("phase2", {"seed": 4, "grasp": "g1"})
    reordered = stable_trial_id("phase2", {"grasp": "g1", "seed": 4})
    other = stable_trial_id("phase2", {"seed": 5, "grasp": "g1"})
    assert first == reordered
    assert first != other


def test_incremental_store_resumes_skips_completed_and_repairs_partial_tail(tmp_path):
    path = tmp_path / "trials.jsonl"
    store = _store(path)
    record = {"trial_id": stable_trial_id("test", {"seed": 1}), "seed": 1}
    assert store.append(record) is True
    assert store.append(record) is False
    with path.open("ab") as stream:
        stream.write(b'{"trial_id":"interrupted"')
    second = {"trial_id": stable_trial_id("test", {"seed": 2}), "seed": 2}
    assert store.append(second) is True
    assert store.records() == [record, second]


def test_incremental_store_serializes_parallel_appends(tmp_path):
    path = tmp_path / "parallel.jsonl"
    records = [
        {"trial_id": stable_trial_id("parallel", {"seed": seed}), "seed": seed}
        for seed in range(16)
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        written = list(executor.map(_store(path).append, records))
    assert all(written)
    persisted = _store(path).records()
    assert {record["trial_id"] for record in persisted} == {record["trial_id"] for record in records}
