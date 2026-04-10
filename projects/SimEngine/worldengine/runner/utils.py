import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List
from typing import Optional

import numpy as np

from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
from nuplan.planning.training.callbacks.profile_callback import ProfileCallback
from worldengine.utils.multithreading.worker_pool import WorkerPool

@dataclass
class CommonBuilder:
    """Common builder data."""

    worker: WorkerPool
    multi_main_callback: MultiMainCallback
    output_dir: Path
    profiler: ProfileCallback

@dataclass(frozen=True)
class PlannerReport:
    """
    Information about planner runtimes, etc. to store to disk.
    """

    compute_trajectory_runtimes: List[float]  # time series of compute_trajectory invocation runtimes [s]

    def compute_summary_statistics(self) -> Dict[str, float]:
        """
        Compute summary statistics over report fields.
        :return: dictionary containing summary statistics of each field.
        """
        summary = {}
        for field in fields(self):
            attr_value = getattr(self, field.name)
            # Compute summary stats for each field. They are all lists of floats, defined in PlannerReport.
            summary[f"{field.name}_mean"] = np.mean(attr_value)
            summary[f"{field.name}_median"] = np.median(attr_value)
            summary[f"{field.name}_std"] = np.std(attr_value)

        return summary

@dataclass
class RunnerReport:
    """Report for a runner."""

    succeeded: bool  # True if simulation was successful
    error_message: Optional[str]  # None if simulation succeeded, traceback if it failed
    start_time: float  # Time simulation.run() was called
    end_time: Optional[float]  # Time simulation.run() returned, when the error was logged, or None temporarily
    planner_report: Optional[PlannerReport]  # Planner report containing stats about planner runtime, None if the
    # runner didn't run a planner (eg. MetricRunner), or when a run fails

    # Metadata about the simulations
    scenario_name: str
    planner_name: str
    log_name: str