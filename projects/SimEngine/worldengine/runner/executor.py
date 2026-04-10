import concurrent.futures
import json
import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from omegaconf import DictConfig

from nuplan.common.utils.file_backed_barrier import distributed_sync
from worldengine.utils.multithreading.worker_pool import Task, WorkerPool
from worldengine.envs.base_env import BaseEnv
from worldengine.runner.utils import RunnerReport

logger = logging.getLogger(__name__)


def save_runner_reports(reports: List[RunnerReport], output_dir: Path, report_file: str) -> Path:
    """
    Save runner reports to a JSON file.
    :param reports: List of RunnerReport from all simulations.
    :param output_dir: Directory to save the report file.
    :param report_file: Name of the report file (extension will be replaced with .json).
    :return: Path to the saved report file.
    """
    rows = []
    for report in reports:
        rows.append({
            'scenario_name': report.scenario_name,
            'log_name': report.log_name,
            'planner_name': report.planner_name,
            'succeeded': report.succeeded,
            'duration_s': round(report.end_time - report.start_time, 3) if report.end_time else None,
            'error_message': report.error_message,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / Path(report_file).with_suffix('.json')
    with open(report_path, 'w') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved runner report ({len(reports)} entries) to {report_path}")
    return report_path

def run_simulation(sim_env: BaseEnv, exit_on_failure: bool = False) -> List[RunnerReport]:
    """
    Proxy for calling simulation.
    :param sim_env: An environment which will execute all batched simulations.
    :param exit_on_failure: If true, raises an exception when the simulation fails.
    :return report for the simulation.
    """
    reports = sim_env.run()

    # Create worker-specific completion flag file
    # The config is resolved inside the worker's lazy_init(), so output_dir has the correct worker-specific path
    flag_file_path = Path(sim_env.config.output_dir) / "simulation_completed.flag"
    flag_file_path.parent.mkdir(parents=True, exist_ok=True)
    flag_file_path.touch()
    logger.info(f"Generated completion flag file: {flag_file_path}")

    return reports

def run_runners(envs: List[BaseEnv], worker: Union[WorkerPool, None], cfg: DictConfig) -> None:
    """
    Run a list of runners.
    :param envs: A list of envs.
    :param worker: Worker.
    :param cfg: Hydra config.
    """
    assert len(envs) > 0, 'No environments found to simulate!'

    logger.info('Executing runners...')
    # Start simulations
    number_of_sims = len(envs)

    if worker is None:
        logger.info('No worker found, running simulations in the SINGLE_NODE mode.')
        reports = [run_simulation(env, cfg.exit_on_failure) for env in envs]
    else:
        logger.info(f"Starting {number_of_sims} simulations using {worker.__class__.__name__}!")
        reports: List[List[RunnerReport]] = worker.map(
            Task(fn=run_simulation,
                num_gpus=cfg.number_of_gpus_allocated_per_simulation,
                num_cpus=cfg.number_of_cpus_allocated_per_simulation,
                ),
            envs
        )
    # Flatten the list of lists
    reports = [report for sublist in reports for report in sublist]

    # Store the results in a dictionary so we can easily store error tracebacks in the next step, if needed
    results: Dict[Tuple[str, str, str], RunnerReport] = {
        (report.scenario_name, report.planner_name, report.log_name): report for report in reports
    }

    # Notify user about the result of simulations
    failed_simulations = str()
    number_of_successful = 0
    number_of_failures = 0
    runner_reports: List[RunnerReport] = list(results.values())
    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            if result.error_message is not None:
                number_of_failures += 1
                logger.warning("Failed Simulation.\n '%s'", result.error_message)
                failed_simulations += f"[{result.log_name}, {result.scenario_name}] \n"

    logger.info(f"Number of successful simulations: {number_of_successful}")
    logger.info(f"Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    if number_of_failures > 0:
        logger.info(f"Failed simulations [log, token]:\n{failed_simulations}")

    logger.info('Finished executing runners!')

    save_runner_reports(runner_reports, Path(cfg.output_dir), cfg.runner_report_file)

    # Sync up nodes when running distributed simulation
    distributed_sync(Path(cfg.output_dir / Path("barrier")), cfg.distributed_timeout_seconds)
