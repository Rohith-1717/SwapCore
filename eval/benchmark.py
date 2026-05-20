import argparse
import csv
import sys
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect.tracer import (
    WRITE,
    dataset_suite,
    generate_workload as generate_trace_workload,
    load_trace_npz,
    save_trace_npz,
)


FAULT_LATENCY_NS = 35_000
SWAP_READ_LATENCY_NS = 80_000
SWAP_WRITE_LATENCY_NS = 95_000
DEFAULT_BENCH_ACCESSES = 1_000_000


@dataclass
class BenchmarkMetrics:
    workload: str
    policy: str
    accesses: int
    frames: int
    page_faults: int
    swap_reads: int
    swap_writes: int
    throughput_accesses_per_sec: float
    mean_fault_latency_ns: float
    p99_fault_latency_ns: float

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


class ClockState:
    def __init__(self):
        self.hand = 0
        self.pages: List[int] = []
        self.ref_bits: Dict[int, int] = {}

    def touch(self, vpn: int):
        self.ref_bits[vpn] = 1

    def add(self, vpn: int):
        self.pages.append(vpn)
        self.ref_bits[vpn] = 1

    def evict(self) -> int:
        while True:
            page = self.pages[self.hand]
            if self.ref_bits.get(page, 0) == 0:
                victim = page
                self.pages.pop(self.hand)
                if self.pages:
                    self.hand %= len(self.pages)
                else:
                    self.hand = 0
                self.ref_bits.pop(victim, None)
                return victim
            self.ref_bits[page] = 0
            self.hand = (self.hand + 1) % len(self.pages)


def simulate_policy(trace: Dict[str, np.ndarray], frames: int, policy: str, workload: str) -> BenchmarkMetrics:
    resident = set()
    dirty = set()
    ever_loaded = set()
    lru = OrderedDict()
    clock = ClockState()
    fault_latencies: List[int] = []
    page_faults = 0
    swap_reads = 0
    swap_writes = 0

    start = time.perf_counter()
    vpns = trace["vpn"]
    access_types = trace["access"] if "access" in trace else trace["access_type"]
    future_positions = build_future_positions(vpns) if policy == "learned" else None

    for index, raw_vpn in enumerate(vpns):
        vpn = int(raw_vpn)
        is_write = int(access_types[index]) == WRITE
        if future_positions is not None:
            future_positions[vpn].popleft()

        if vpn in resident:
            if policy == "lru":
                lru.move_to_end(vpn)
            elif policy == "clock":
                clock.touch(vpn)
            if is_write:
                dirty.add(vpn)
            continue

        page_faults += 1
        latency = FAULT_LATENCY_NS
        if vpn in ever_loaded:
            swap_reads += 1
            latency += SWAP_READ_LATENCY_NS

        if len(resident) >= frames:
            victim = select_victim(policy, resident, lru, clock, future_positions)
            resident.remove(victim)
            if victim in dirty:
                dirty.remove(victim)
                swap_writes += 1
                latency += SWAP_WRITE_LATENCY_NS

        resident.add(vpn)
        ever_loaded.add(vpn)
        if is_write:
            dirty.add(vpn)
        if policy == "lru":
            lru[vpn] = None
        elif policy == "clock":
            clock.add(vpn)
        fault_latencies.append(latency)

    elapsed = max(time.perf_counter() - start, 1e-9)
    latencies = np.array(fault_latencies or [0], dtype=np.float64)
    return BenchmarkMetrics(
        workload=workload,
        policy=policy,
        accesses=len(vpns),
        frames=frames,
        page_faults=page_faults,
        swap_reads=swap_reads,
        swap_writes=swap_writes,
        throughput_accesses_per_sec=len(vpns) / elapsed,
        mean_fault_latency_ns=float(np.mean(latencies)),
        p99_fault_latency_ns=float(np.percentile(latencies, 99)),
    )


def select_victim(
    policy: str,
    resident: set,
    lru: OrderedDict,
    clock: ClockState,
    future_positions: Optional[Dict[int, deque]],
) -> int:
    if policy == "lru":
        victim = next(iter(lru))
        lru.pop(victim, None)
        return victim
    if policy == "clock":
        return clock.evict()
    if policy == "learned":
        return reuse_hint_victim(resident, future_positions)
    raise ValueError(f"unknown policy: {policy}")


def build_future_positions(vpns: np.ndarray) -> Dict[int, deque]:
    positions = defaultdict(deque)
    for index, raw_vpn in enumerate(vpns):
        positions[int(raw_vpn)].append(index)
    return positions


def reuse_hint_victim(resident: Iterable[int], future_positions: Optional[Dict[int, deque]]) -> int:
    """Deterministic learned-policy placeholder using trace reuse hints.

    Runtime GRU integration can replace this scorer with exported-model inference.
    For trace-generation work, it gives a stable upper-bound-ish policy that
    prefers evicting pages with no near reuse.
    """
    if future_positions is None:
        return next(iter(resident))
    return max(
        resident,
        key=lambda page: future_positions[page][0] if future_positions[page] else 1_000_000_000_000,
    )


def generate_workload(name: str, accesses: int, num_pages: int, seed: int) -> Dict[str, np.ndarray]:
    return generate_trace_workload(name, accesses, num_pages, seed)


def run_benchmark(args) -> List[BenchmarkMetrics]:
    if args.trace:
        trace = load_trace_npz(args.trace)
        traces = {Path(args.trace).stem: trace}
    elif args.suite:
        if args.accesses == DEFAULT_BENCH_ACCESSES:
            traces = dataset_suite(args.num_pages, args.seed)
        else:
            traces = {
                "sequential": generate_workload("sequential", args.accesses, args.num_pages, args.seed + 10),
                "random": generate_workload("random", args.accesses, args.num_pages, args.seed + 20),
                "loop": generate_workload("loop", args.accesses, args.num_pages, args.seed + 30),
                "zipfian": generate_workload("zipfian", args.accesses, args.num_pages, args.seed + 40),
                "mixed": generate_workload("mixed", args.accesses, args.num_pages, args.seed + 50),
            }
    else:
        traces = {args.workload: generate_workload(args.workload, args.accesses, args.num_pages, args.seed)}

    if args.save_traces:
        output_dir = Path(args.save_traces)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, trace in traces.items():
            save_trace_npz(trace, output_dir / f"{name}.npz")

    results = []
    for workload, trace in traces.items():
        for policy in args.policies:
            results.append(simulate_policy(trace, args.frames, policy, workload))
    return results


def write_csv(results: List[BenchmarkMetrics], path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].as_dict().keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_dict())


def print_results(results: List[BenchmarkMetrics]):
    header = (
        "workload",
        "policy",
        "accesses",
        "faults",
        "reads",
        "writes",
        "throughput/s",
        "p99 ns",
    )
    print(",".join(header))
    for result in results:
        print(
            f"{result.workload},{result.policy},{result.accesses},{result.page_faults},"
            f"{result.swap_reads},{result.swap_writes},"
            f"{result.throughput_accesses_per_sec:.0f},{result.p99_fault_latency_ns:.0f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate traces and benchmark SwapCore eviction policies.")
    parser.add_argument("--workload", choices=["sequential", "random", "zipfian", "mixed"], default="mixed")
    parser.add_argument("--trace", help="Replay an existing .npz trace.")
    parser.add_argument("--suite", action="store_true", help="Run the full deterministic workload suite.")
    parser.add_argument("--accesses", type=int, default=DEFAULT_BENCH_ACCESSES)
    parser.add_argument("--num-pages", type=int, default=65536)
    parser.add_argument("--frames", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policies", nargs="+", default=["lru", "clock", "learned"], choices=["lru", "clock", "learned"])
    parser.add_argument("--save-traces", help="Directory for generated .npz traces.")
    parser.add_argument("--csv", help="Write benchmark metrics to CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_benchmark(args)
    print_results(results)
    if args.csv:
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
