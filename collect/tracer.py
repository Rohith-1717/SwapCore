import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


READ = 0
WRITE = 1
ACCESS_TYPE_NAMES = np.array(["read", "write"])
TRACE_KEYS = ("vpn", "ts", "delta", "access", "phase")


@dataclass
class TraceRecord:
    vpn: int
    access_delta: int
    access_type: str
    timestamp: int
    reuse_distance: int
    future_reuse: Optional[bool] = None
    working_set: int = 0
    phase: int = 0
    access_size: int = 4096


class TraceCollector:
    """Small compatibility collector for tests and hand-built traces."""

    def __init__(self):
        self.records: List[TraceRecord] = []

    def add_access(
        self,
        vpn: int,
        access_type: str,
        timestamp: int,
        access_delta: int,
        reuse_distance: int,
        future_reuse: Optional[bool] = None,
        working_set: int = 0,
        phase: int = 0,
        access_size: int = 4096,
    ):
        self.records.append(
            TraceRecord(
                vpn,
                access_delta,
                access_type,
                timestamp,
                reuse_distance,
                future_reuse,
                working_set,
                phase,
                access_size,
            )
        )

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return records_to_arrays(self.records)

    def save(self, path: str):
        path_obj = Path(path)
        if path_obj.suffix == ".npz":
            save_trace_npz(self.to_arrays(), path_obj)
            return

        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as file:
            json.dump([asdict(record) for record in self.records], file, indent=2)

    @staticmethod
    def load(path: str):
        path_obj = Path(path)
        if path_obj.suffix == ".npz":
            arrays = load_trace_npz(path_obj)
            collector = TraceCollector()
            access_types = ACCESS_TYPE_NAMES[arrays["access"]]
            for index in range(len(arrays["vpn"])):
                collector.add_access(
                    int(arrays["vpn"][index]),
                    str(access_types[index]),
                    int(arrays["ts"][index]),
                    int(arrays["delta"][index]),
                    int(arrays["delta"][index]),
                    None,
                    0,
                    int(arrays["phase"][index]),
                    4096,
                )
            return collector

        with path_obj.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        collector = TraceCollector()
        for item in raw:
            collector.records.append(TraceRecord(**item))
        return collector


def records_to_arrays(records: Sequence[TraceRecord]) -> Dict[str, np.ndarray]:
    size = len(records)
    vpn = np.empty(size, dtype=np.int32)
    ts = np.empty(size, dtype=np.int64)
    delta = np.empty(size, dtype=np.int32)
    access = np.empty(size, dtype=np.int8)
    phase = np.empty(size, dtype=np.int8)
    for index, record in enumerate(records):
        vpn[index] = record.vpn
        ts[index] = record.timestamp
        delta[index] = record.access_delta
        access[index] = WRITE if record.access_type == "write" else READ
        phase[index] = record.phase
    return finalize_trace(vpn, access, phase, ts=ts, delta=delta)


def save_trace_npz(trace: Dict[str, np.ndarray], path: str | Path):
    trace = normalize_trace(trace)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vpn=trace["vpn"],
        ts=trace["ts"],
        delta=trace["delta"],
        access=trace["access"],
        phase=trace["phase"],
    )


def load_trace_npz(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return normalize_trace({name: data[name] for name in data.files})


def normalize_trace(trace: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    vpn = trace["vpn"]
    ts = trace["ts"] if "ts" in trace else trace["timestamp"]
    delta = trace["delta"] if "delta" in trace else trace["access_delta"]
    access = trace["access"] if "access" in trace else trace["access_type"]
    phase = trace["phase"] if "phase" in trace else np.zeros(len(vpn), dtype=np.int8)

    if access.dtype.kind in {"U", "S", "O"}:
        access = np.array([WRITE if value == "write" else READ for value in access], dtype=np.int8)

    return {
        "vpn": np.ascontiguousarray(vpn, dtype=np.int32),
        "ts": np.ascontiguousarray(ts, dtype=np.int64),
        "delta": np.ascontiguousarray(np.clip(delta, 0, np.iinfo(np.int32).max), dtype=np.int32),
        "access": np.ascontiguousarray(access, dtype=np.int8),
        "phase": np.ascontiguousarray(phase, dtype=np.int8),
    }


def sequential_scan(
    num_pages: int,
    accesses: int,
    seed: int = 0,
    write_ratio: float = 0.05,
    start_page: int = 0,
    phase: int = 0,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    vpn = start_page + (np.arange(accesses, dtype=np.int32) % np.int32(num_pages))
    delta = np.zeros(accesses, dtype=np.int32)
    if accesses > num_pages:
        delta[num_pages:] = num_pages
    return finalize_trace(vpn, access_types(accesses, rng, write_ratio), full_phase(accesses, phase), delta=delta)


def random_access(
    num_pages: int,
    accesses: int,
    seed: int = 0,
    write_ratio: float = 0.25,
    start_page: int = 0,
    phase: int = 1,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    vpn = start_page + rng.integers(0, num_pages, size=accesses, dtype=np.int32)
    return finalize_trace(vpn, access_types(accesses, rng, write_ratio), full_phase(accesses, phase))


def looping_working_set(
    working_set_pages: int,
    accesses: int,
    seed: int = 0,
    write_ratio: float = 0.15,
    start_page: int = 0,
    phase: int = 2,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    vpn = start_page + (np.arange(accesses, dtype=np.int32) % np.int32(working_set_pages))
    delta = np.zeros(accesses, dtype=np.int32)
    if accesses > working_set_pages:
        delta[working_set_pages:] = working_set_pages
    return finalize_trace(vpn, access_types(accesses, rng, write_ratio), full_phase(accesses, phase), delta=delta)


def zipfian_access(
    num_pages: int,
    accesses: int,
    seed: int = 0,
    exponent: float = 1.2,
    write_ratio: float = 0.20,
    start_page: int = 0,
    phase: int = 3,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ranks = rng.zipf(exponent, size=accesses)
    vpn = start_page + ((ranks - 1) % num_pages).astype(np.int32)
    return finalize_trace(vpn, access_types(accesses, rng, write_ratio), full_phase(accesses, phase))


def mixed_phase_workload(num_pages: int, accesses: int, seed: int = 0) -> Dict[str, np.ndarray]:
    phase_size = 1_000_000
    chunks = []
    remaining = accesses
    cycle = 0
    while remaining > 0:
        chunk_size = min(phase_size, remaining)
        phase = cycle % 4
        if phase == 0:
            chunk = sequential_scan(num_pages, chunk_size, seed + cycle, 0.05, 0, phase)
        elif phase == 1:
            chunk = random_access(num_pages, chunk_size, seed + cycle, 0.30, 0, phase)
        elif phase == 2:
            loop_pages = max(32, min(num_pages // 128, 2048))
            chunk = looping_working_set(loop_pages, chunk_size, seed + cycle, 0.18, 0, phase)
        else:
            zipf_pages = max(1024, num_pages // 4)
            chunk = zipfian_access(zipf_pages, chunk_size, seed + cycle, 1.15, 0.22, 0, phase)
        chunks.append(chunk)
        remaining -= chunk_size
        cycle += 1
    return concatenate_traces(chunks)


def dataset_suite(num_pages: int = 65536, seed: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        "sequential_5m": sequential_scan(num_pages, 5_000_000, seed + 10),
        "random_5m": random_access(num_pages, 5_000_000, seed + 20),
        "loop_5m": looping_working_set(max(32, num_pages // 128), 5_000_000, seed + 30),
        "zipfian_5m": zipfian_access(num_pages, 5_000_000, seed + 40),
        "mixed_10m": mixed_phase_workload(num_pages, 10_000_000, seed + 50),
    }


def finalize_trace(
    vpn: np.ndarray,
    access: np.ndarray,
    phase: np.ndarray,
    ts: Optional[np.ndarray] = None,
    delta: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    count = len(vpn)
    if ts is None:
        ts = np.arange(count, dtype=np.int64)
    if delta is None:
        delta = compute_access_delta(vpn)
    return normalize_trace({"vpn": vpn, "ts": ts, "delta": delta, "access": access, "phase": phase})


def concatenate_traces(traces: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    traces = [normalize_trace(trace) for trace in traces]
    combined = {
        key: np.ascontiguousarray(np.concatenate([trace[key] for trace in traces]))
        for key in TRACE_KEYS
    }
    combined["ts"] = np.arange(len(combined["vpn"]), dtype=np.int64)
    combined["delta"] = compute_access_delta(combined["vpn"])
    return normalize_trace(combined)


def access_types(count: int, rng: np.random.Generator, write_ratio: float) -> np.ndarray:
    return np.ascontiguousarray((rng.random(count) < write_ratio).astype(np.int8))


def full_phase(count: int, phase: int) -> np.ndarray:
    return np.full(count, phase, dtype=np.int8)


def compute_access_delta(vpn: np.ndarray) -> np.ndarray:
    deltas = np.zeros(len(vpn), dtype=np.int64)
    if len(vpn) == 0:
        return deltas.astype(np.int32)

    order = np.argsort(vpn, kind="stable")
    ordered_vpns = vpn[order]
    same_page = ordered_vpns[1:] == ordered_vpns[:-1]
    current_positions = order[1:][same_page]
    previous_positions = order[:-1][same_page]
    deltas[current_positions] = current_positions - previous_positions
    return np.ascontiguousarray(np.clip(deltas, 0, np.iinfo(np.int32).max), dtype=np.int32)


def print_stats(name: str, trace: Dict[str, np.ndarray]):
    trace = normalize_trace(trace)
    phases, counts = np.unique(trace["phase"], return_counts=True)
    phase_counts = {int(phase): int(count) for phase, count in zip(phases, counts)}
    print(f"{name}:")
    print(f"  shape: vpn={trace['vpn'].shape}, ts={trace['ts'].shape}, delta={trace['delta'].shape}, access={trace['access'].shape}, phase={trace['phase'].shape}")
    print(f"  dtype: vpn={trace['vpn'].dtype}, ts={trace['ts'].dtype}, delta={trace['delta'].dtype}, access={trace['access'].dtype}, phase={trace['phase'].dtype}")
    print(f"  unique_vpns: {np.unique(trace['vpn']).size}")
    print(f"  phase_distribution: {phase_counts}")


def generate_workload(name: str, accesses: int, num_pages: int, seed: int) -> Dict[str, np.ndarray]:
    if name == "sequential":
        return sequential_scan(num_pages, accesses, seed)
    if name == "random":
        return random_access(num_pages, accesses, seed)
    if name == "loop":
        return looping_working_set(max(32, num_pages // 128), accesses, seed)
    if name == "zipfian":
        return zipfian_access(num_pages, accesses, seed)
    if name == "mixed":
        return mixed_phase_workload(num_pages, accesses, seed)
    raise ValueError(f"unknown workload: {name}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate deterministic SwapCore memory traces.")
    parser.add_argument("--workload", choices=["sequential", "random", "loop", "zipfian", "mixed", "suite"], default="mixed")
    parser.add_argument("--output", default="data/traces/mixed_10m.npz")
    parser.add_argument("--num-pages", type=int, default=65536)
    parser.add_argument("--accesses", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.workload == "suite":
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, trace in dataset_suite(args.num_pages, args.seed).items():
            out_path = output_dir / f"{name}.npz"
            save_trace_npz(trace, out_path)
            print(f"saved {out_path}")
            print_stats(name, trace)
        return

    trace = generate_workload(args.workload, args.accesses, args.num_pages, args.seed)
    save_trace_npz(trace, args.output)
    print(f"saved {args.output}")
    print_stats(args.workload, trace)


if __name__ == "__main__":
    main()
