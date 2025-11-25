#!/usr/bin/env python3
"""
MPK Runtime Concepts Example

This script demonstrates the key concepts of Mirage's Persistent Kernel (MPK) Runtime:
1. Task Graph representation
2. Event-driven execution simulation
3. Inter-layer pipelining visualization
4. Worker/Scheduler model

Note: This is a conceptual demonstration. The actual runtime is implemented in CUDA.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Callable
from enum import Enum
from collections import deque
import time


class TaskType(Enum):
    RMSNORM = "RMSNorm"
    ATTENTION = "Attention"
    FFN_GATE = "FFN_Gate"
    FFN_UP = "FFN_Up"
    FFN_DOWN = "FFN_Down"
    LINEAR = "Linear"
    COMM_SEND = "Comm_Send"
    COMM_RECV = "Comm_Recv"


class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass
class Task:
    """Represents a task in the MPK task graph."""
    task_id: int
    task_type: TaskType
    layer_idx: int
    token_idx: int
    dependencies: List[int] = field(default_factory=list)
    dependents: List[int] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    execution_time: float = 0.1  # Simulated execution time
    sm_id: int = -1  # Which SM executed this task


@dataclass
class Event:
    """Represents a task completion event."""
    task_id: int
    timestamp: float


class TaskQueue:
    """Thread-safe task queue (simulated)."""

    def __init__(self):
        self.queue: deque = deque()
        self.lock_count = 0

    def enqueue(self, task: Task):
        task.status = TaskStatus.READY
        self.queue.append(task)

    def dequeue(self) -> Optional[Task]:
        if self.queue:
            return self.queue.popleft()
        return None

    def __len__(self):
        return len(self.queue)


class EventQueue:
    """Event queue for task completion notifications."""

    def __init__(self):
        self.events: List[Event] = []

    def post(self, task_id: int, timestamp: float):
        self.events.append(Event(task_id, timestamp))

    def get_events(self) -> List[Event]:
        events = self.events.copy()
        self.events.clear()
        return events


class Worker:
    """Simulates an SM (Streaming Multiprocessor) as a worker."""

    def __init__(self, sm_id: int):
        self.sm_id = sm_id
        self.current_task: Optional[Task] = None
        self.busy_until: float = 0.0
        self.tasks_completed: List[int] = []

    def is_busy(self, current_time: float) -> bool:
        return current_time < self.busy_until

    def assign_task(self, task: Task, current_time: float):
        self.current_task = task
        task.status = TaskStatus.RUNNING
        task.sm_id = self.sm_id
        self.busy_until = current_time + task.execution_time

    def finish_task(self) -> Optional[Task]:
        if self.current_task:
            task = self.current_task
            task.status = TaskStatus.COMPLETED
            self.tasks_completed.append(task.task_id)
            self.current_task = None
            return task
        return None


class MPKRuntime:
    """Simulates the MPK runtime system."""

    def __init__(self, num_sms: int = 4, num_schedulers: int = 1):
        self.num_sms = num_sms
        self.workers = [Worker(i) for i in range(num_sms)]
        self.task_queue = TaskQueue()
        self.event_queue = EventQueue()
        self.tasks: Dict[int, Task] = {}
        self.deps_remaining: Dict[int, int] = {}
        self.current_time: float = 0.0
        self.execution_log: List[tuple] = []

    def add_task(self, task: Task):
        self.tasks[task.task_id] = task
        self.deps_remaining[task.task_id] = len(task.dependencies)

    def build_dependents(self):
        """Build reverse dependency graph."""
        for task in self.tasks.values():
            for dep_id in task.dependencies:
                self.tasks[dep_id].dependents.append(task.task_id)

    def run(self) -> float:
        """Run the simulation and return total execution time."""
        self.build_dependents()

        # Initialize: enqueue tasks with no dependencies
        for task in self.tasks.values():
            if not task.dependencies:
                self.task_queue.enqueue(task)

        while True:
            # Check for finished tasks
            for worker in self.workers:
                if worker.current_task and not worker.is_busy(self.current_time):
                    finished_task = worker.finish_task()
                    if finished_task:
                        self.event_queue.post(finished_task.task_id, self.current_time)
                        self.execution_log.append((
                            finished_task.task_id,
                            finished_task.task_type.value,
                            finished_task.layer_idx,
                            finished_task.token_idx,
                            worker.sm_id,
                            self.current_time - finished_task.execution_time,
                            self.current_time
                        ))

            # Process events: check if dependent tasks are ready
            events = self.event_queue.get_events()
            for event in events:
                completed_task = self.tasks[event.task_id]
                for dep_id in completed_task.dependents:
                    self.deps_remaining[dep_id] -= 1
                    if self.deps_remaining[dep_id] == 0:
                        self.task_queue.enqueue(self.tasks[dep_id])

            # Assign tasks to idle workers
            for worker in self.workers:
                if not worker.is_busy(self.current_time) and len(self.task_queue) > 0:
                    task = self.task_queue.dequeue()
                    if task:
                        worker.assign_task(task, self.current_time)

            # Check termination
            all_done = all(t.status == TaskStatus.COMPLETED for t in self.tasks.values())
            all_idle = all(not w.is_busy(self.current_time) for w in self.workers)

            if all_done or (all_idle and len(self.task_queue) == 0):
                break

            # Advance time
            next_finish_times = [w.busy_until for w in self.workers if w.is_busy(self.current_time)]
            if next_finish_times:
                self.current_time = min(next_finish_times)
            else:
                self.current_time += 0.01

        return self.current_time


def create_llm_layer_tasks(layer_idx: int, token_idx: int, base_id: int,
                            prev_layer_tasks: List[int]) -> List[Task]:
    """Create tasks for one transformer layer."""
    tasks = []

    # RMSNorm
    rmsnorm = Task(
        task_id=base_id,
        task_type=TaskType.RMSNORM,
        layer_idx=layer_idx,
        token_idx=token_idx,
        dependencies=prev_layer_tasks,
        execution_time=0.05
    )
    tasks.append(rmsnorm)

    # Attention (depends on RMSNorm)
    attention = Task(
        task_id=base_id + 1,
        task_type=TaskType.ATTENTION,
        layer_idx=layer_idx,
        token_idx=token_idx,
        dependencies=[base_id],
        execution_time=0.15
    )
    tasks.append(attention)

    # FFN Gate and Up (can run in parallel, depend on Attention)
    ffn_gate = Task(
        task_id=base_id + 2,
        task_type=TaskType.FFN_GATE,
        layer_idx=layer_idx,
        token_idx=token_idx,
        dependencies=[base_id + 1],
        execution_time=0.08
    )
    tasks.append(ffn_gate)

    ffn_up = Task(
        task_id=base_id + 3,
        task_type=TaskType.FFN_UP,
        layer_idx=layer_idx,
        token_idx=token_idx,
        dependencies=[base_id + 1],
        execution_time=0.08
    )
    tasks.append(ffn_up)

    # FFN Down (depends on both Gate and Up)
    ffn_down = Task(
        task_id=base_id + 4,
        task_type=TaskType.FFN_DOWN,
        layer_idx=layer_idx,
        token_idx=token_idx,
        dependencies=[base_id + 2, base_id + 3],
        execution_time=0.08
    )
    tasks.append(ffn_down)

    return tasks


def visualize_execution(execution_log: List[tuple], num_sms: int, title: str):
    """Visualize task execution timeline."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    if not execution_log:
        print("  No tasks executed.")
        return

    max_time = max(end for _, _, _, _, _, _, end in execution_log)
    time_scale = 50 / max_time if max_time > 0 else 1

    print(f"\n  Time scale: 1 char = {max_time/50:.3f} time units")
    print(f"  Total execution time: {max_time:.3f}")
    print()

    # Group by SM
    sm_tasks = {i: [] for i in range(num_sms)}
    for task_id, task_type, layer, token, sm_id, start, end in execution_log:
        sm_tasks[sm_id].append((start, end, f"L{layer}T{token}:{task_type[:3]}"))

    # Print timeline for each SM
    for sm_id in range(num_sms):
        timeline = [' '] * 50
        for start, end, label in sm_tasks[sm_id]:
            start_pos = int(start * time_scale)
            end_pos = int(end * time_scale)
            # Fill timeline
            char = label[0] if label else '#'
            for i in range(start_pos, min(end_pos, 50)):
                timeline[i] = char

        print(f"  SM {sm_id}: |{''.join(timeline)}|")

    print(f"        0{' '*23}Time{' '*20}{max_time:.2f}")


def example_single_token():
    """Example: Processing a single token through multiple layers."""
    print("\n" + "="*70)
    print("Example 1: Single Token Through 3 Layers (Traditional Sequential)")
    print("="*70)
    print("""
    Without pipelining, each layer must complete before the next starts.
    This demonstrates the baseline sequential execution.
    """)

    runtime = MPKRuntime(num_sms=4)
    task_id = 0

    prev_deps = []
    for layer in range(3):
        layer_tasks = create_llm_layer_tasks(layer, token_idx=0, base_id=task_id,
                                              prev_layer_tasks=prev_deps)
        for task in layer_tasks:
            runtime.add_task(task)
        # Last task of layer is FFN_Down
        prev_deps = [task_id + 4]
        task_id += 5

    total_time = runtime.run()
    visualize_execution(runtime.execution_log, runtime.num_sms, "Single Token Execution")
    print(f"\n  Total time: {total_time:.3f} units")


def example_pipelined_tokens():
    """Example: Processing multiple tokens with inter-layer pipelining."""
    print("\n" + "="*70)
    print("Example 2: Multiple Tokens with Inter-Layer Pipelining (MPK)")
    print("="*70)
    print("""
    With MPK, different tokens can be at different layers simultaneously.
    Token 1 can start Layer 0 while Token 0 is in Layer 1.

    This overlapping execution significantly improves throughput.
    """)

    runtime = MPKRuntime(num_sms=4)
    task_id = 0
    num_tokens = 3
    num_layers = 3

    # Create tasks for all tokens across all layers
    layer_last_tasks: Dict[tuple, List[int]] = {}  # (layer, token) -> [last_task_ids]

    for token in range(num_tokens):
        for layer in range(num_layers):
            # Dependencies: same token's previous layer, OR previous token's same layer
            deps = []

            # Dependency on previous layer of same token
            if layer > 0:
                deps.extend(layer_last_tasks.get((layer - 1, token), []))

            # Note: In true pipelining, we don't wait for previous token
            # This allows overlap!

            layer_tasks = create_llm_layer_tasks(
                layer, token_idx=token, base_id=task_id,
                prev_layer_tasks=deps
            )
            for task in layer_tasks:
                runtime.add_task(task)

            layer_last_tasks[(layer, token)] = [task_id + 4]  # FFN_Down
            task_id += 5

    total_time = runtime.run()
    visualize_execution(runtime.execution_log, runtime.num_sms, "Pipelined Token Execution")
    print(f"\n  Total time: {total_time:.3f} units")

    # Compare with sequential
    seq_time = 0.44 * num_tokens * num_layers / num_tokens  # Approximate
    print(f"  Speedup from pipelining: ~{seq_time * num_tokens / total_time:.2f}x")


def example_multi_gpu_overlap():
    """Example: Computation and communication overlap for multi-GPU."""
    print("\n" + "="*70)
    print("Example 3: Multi-GPU Compute/Communication Overlap")
    print("="*70)
    print("""
    In multi-GPU inference, each GPU processes some layers.
    Communication (tensor transfer) can overlap with computation.

    GPU 0: [Compute L0] [Send] [Compute L2] [Send]
    GPU 1:      [Recv] [Compute L1] [Recv] [Compute L3]

    The MPK runtime automatically schedules these overlapping operations.
    """)

    runtime = MPKRuntime(num_sms=4)

    # Simplified multi-GPU task graph
    tasks = [
        Task(0, TaskType.RMSNORM, 0, 0, [], execution_time=0.1),      # L0 compute
        Task(1, TaskType.ATTENTION, 0, 0, [0], execution_time=0.15),
        Task(2, TaskType.COMM_SEND, 0, 0, [1], execution_time=0.05),  # Send to GPU 1
        Task(3, TaskType.COMM_RECV, 1, 0, [], execution_time=0.05),   # Recv on GPU 1 (parallel with send)
        Task(4, TaskType.RMSNORM, 1, 0, [3], execution_time=0.1),     # L1 compute (after recv)
        Task(5, TaskType.ATTENTION, 1, 0, [4], execution_time=0.15),
        Task(6, TaskType.RMSNORM, 2, 0, [2], execution_time=0.1),     # L2 compute (after send, GPU 0)
        Task(7, TaskType.ATTENTION, 2, 0, [6], execution_time=0.15),
        Task(8, TaskType.COMM_SEND, 2, 0, [7], execution_time=0.05),  # Send L2 result
        Task(9, TaskType.COMM_RECV, 3, 0, [5], execution_time=0.05),  # Recv for L3
        Task(10, TaskType.RMSNORM, 3, 0, [9], execution_time=0.1),    # L3 compute
        Task(11, TaskType.ATTENTION, 3, 0, [10], execution_time=0.15),
    ]

    for task in tasks:
        runtime.add_task(task)

    total_time = runtime.run()
    visualize_execution(runtime.execution_log, runtime.num_sms,
                        "Multi-GPU Compute/Comm Overlap")
    print(f"\n  Total time: {total_time:.3f} units")
    print("  Notice: Communication tasks can overlap with computation on different SMs")


def example_task_graph_structure():
    """Visualize the task graph structure."""
    print("\n" + "="*70)
    print("Example 4: Task Graph Structure Visualization")
    print("="*70)
    print("""
    The task graph represents dependencies between operations.
    Each node is a task; edges represent data dependencies.

    For a transformer layer:

    ┌─────────┐
    │ RMSNorm │ (normalize input)
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │  Attn   │ (self-attention)
    └────┬────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
    ┌─────┐ ┌─────┐
    │Gate │ │ Up  │  (FFN parallel paths)
    └──┬──┘ └──┬──┘
       │       │
       └───┬───┘
           │
           ▼
       ┌───────┐
       │ Down  │ (FFN output)
       └───────┘

    Tasks at the same level can potentially execute in parallel on different SMs.
    """)

    # Create and print task info
    tasks = create_llm_layer_tasks(0, 0, 0, [])
    print("\n  Task Details:")
    print("  " + "-"*50)
    for task in tasks:
        deps = task.dependencies if task.dependencies else "none"
        print(f"  Task {task.task_id}: {task.task_type.value:10s} deps={deps}")


def example_worker_scheduler_model():
    """Explain the worker/scheduler model."""
    print("\n" + "="*70)
    print("Example 5: Worker and Scheduler Model")
    print("="*70)
    print("""
    MPK Runtime Architecture:

    ┌─────────────────────────────────────────────────────────┐
    │              Mega-Kernel (Single CUDA Launch)           │
    │                                                         │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │           Scheduler Warps (1-2 per SM)          │   │
    │  │                                                  │   │
    │  │  - Monitor task queue in global memory          │   │
    │  │  - Check dependency satisfaction                │   │
    │  │  - Signal ready tasks to workers                │   │
    │  │  - Handle event queue updates                   │   │
    │  └─────────────────────────────────────────────────┘   │
    │                         │                               │
    │                         ▼                               │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │          Task Queue (Global Memory)             │   │
    │  │                                                  │   │
    │  │  [Ready Task 0] [Ready Task 1] [Ready Task 2]   │   │
    │  └─────────────────────────────────────────────────┘   │
    │                         │                               │
    │                         ▼                               │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │             Worker Warps (per SM)               │   │
    │  │                                                  │   │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐           │   │
    │  │  │Worker 0 │ │Worker 1 │ │Worker 2 │  ...      │   │
    │  │  │ (SM 0)  │ │ (SM 1)  │ │ (SM 2)  │           │   │
    │  │  │         │ │         │ │         │           │   │
    │  │  │ Execute │ │ Execute │ │ Execute │           │   │
    │  │  │ Tasks   │ │ Tasks   │ │ Tasks   │           │   │
    │  │  └─────────┘ └─────────┘ └─────────┘           │   │
    │  └─────────────────────────────────────────────────┘   │
    │                         │                               │
    │                         ▼                               │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │          Event Queue (Global Memory)            │   │
    │  │                                                  │   │
    │  │  [Event: Task 0 done] [Event: Task 3 done] ...  │   │
    │  └─────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘

    Key Points:
    1. Single kernel launch - no host-side kernel launch overhead
    2. Device-side scheduling - tasks dispatched within the kernel
    3. Event-driven - completion events trigger dependent tasks
    4. Load balancing - SMs pull tasks from shared queue
    """)


def main():
    print("""
    ================================================================
    Mirage Persistent Kernel (MPK) Runtime Demonstration
    ================================================================

    This tutorial demonstrates the key concepts of the MPK runtime:

    1. Task Graph: Sub-kernel level representation of computation
    2. Event-Driven Execution: Tasks triggered by dependency completion
    3. Inter-Layer Pipelining: Overlapping execution of different tokens
    4. Compute/Comm Overlap: Hiding communication latency
    5. Worker/Scheduler Model: Device-side task dispatch

    The MPK approach eliminates kernel launch overhead and enables
    fine-grained parallelism that traditional CUDA programming cannot achieve.
    """)

    example_task_graph_structure()
    example_worker_scheduler_model()
    example_single_token()
    example_pipelined_tokens()
    example_multi_gpu_overlap()

    print("\n" + "="*70)
    print("Summary: MPK Runtime Benefits")
    print("="*70)
    print("""
    1. Zero kernel launch overhead (single persistent kernel)
    2. Fine-grained task parallelism (sub-kernel level)
    3. Automatic inter-layer pipelining
    4. Overlapped compute and communication
    5. Dynamic load balancing across SMs

    For implementation details, see:
      - include/mirage/persistent_kernel/
      - python/mirage/persistent_kernel.py
      - The HC2025 presentation PDF in this directory
    """)


if __name__ == "__main__":
    main()
