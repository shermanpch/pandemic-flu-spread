import copy
import logging
import math
import multiprocessing
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import simpy

from simulation import Simulation
from stats import StatisticsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_simulation(args):
    """
    Run a single simulation and return the results.

    Args:
        args (tuple): A tuple containing (simulation_params, run_number).

    Returns:
        dict: A dictionary containing the final results and statistics collector.
    """
    simulation_params, run_number = args

    # Set a unique random seed for this run
    random_seed = simulation_params.get("random_seed", None)
    if random_seed is not None:
        sim_random_seed = random_seed + run_number
    else:
        sim_random_seed = None  # Let random seed be random

    # Create a new SimPy environment for this simulation run
    env = simpy.Environment()

    # Create a StatisticsCollector for this run
    stats_collector = StatisticsCollector()

    # Update simulation parameters with new random seed and environment
    sim_params = copy.deepcopy(simulation_params)
    sim_params["random_seed"] = sim_random_seed
    sim_params["env"] = env
    sim_params["statistics_collector"] = stats_collector

    # Initialize the simulation
    sim = Simulation(**sim_params)

    # Run the simulation
    sim.env.run()

    # Collect key metrics from this run
    final_data = stats_collector.get_final_data()

    return {
        "final_data": final_data,
        "stats_collector": stats_collector,
        "run_number": run_number,
    }


class SimulationBatchRunner:
    """
    Runs multiple simulations and aggregates the results for statistical analysis.

    Attributes:
        num_runs (int): Number of simulation runs to perform.
        simulation_params (dict): Parameters for the simulations.
        stats (List[StatisticsCollector]): Statistics from each simulation run.
        outcomes (List[Dict[str, int]]): Key metrics from each run.
        aggregated_data (Dict[str, List[float]]): Aggregated results across all runs.
        daily_aggregated_data (List[Dict[str, List[float]]]): Aggregated daily data.
        daily_expected_counts (List[Dict[str, Dict[str, float]]]): Expected counts per day.
    """

    def __init__(self, num_runs: int, simulation_params: dict):
        """
        Initialize the SimulationBatchRunner.

        Args:
            num_runs (int): Number of simulation runs to perform.
            simulation_params (dict): Parameters for the simulations.
        """
        self.num_runs = num_runs
        self.simulation_params = simulation_params
        self.stats: List[StatisticsCollector] = []
        self.outcomes: List[Dict[str, int]] = []
        self.aggregated_data: Dict[str, List[float]] = {}
        self.daily_aggregated_data: List[Dict[str, List[float]]] = []
        self.daily_expected_counts: List[Dict[str, Dict[str, float]]] = []

    def run(self) -> None:
        """Run all simulations and aggregate the results using multiprocessing."""
        # Prepare arguments for each simulation run
        simulation_args = [
            (self.simulation_params, run_number) for run_number in range(self.num_runs)
        ]

        # Use multiprocessing to run simulations in parallel
        with multiprocessing.Pool() as pool:
            # Map the function over the arguments
            results = pool.map(run_single_simulation, simulation_args)

        # Process the results
        for result in results:
            final_data = result["final_data"]
            stats_collector = result["stats_collector"]
            run_number = result["run_number"]

            self.stats.append(stats_collector)
            self.outcomes.append(final_data)
            logger.debug(f"Simulation {run_number + 1}/{self.num_runs} completed.")

        # After all runs, aggregate the results
        self._aggregate_results()

        # Aggregate daily statistics
        self._aggregate_daily_stats()

    def _aggregate_results(self) -> None:
        """Aggregate key metrics across all runs."""
        total_infections = [outcome["total_infections"] for outcome in self.outcomes]
        total_deaths = [outcome["total_deaths"] for outcome in self.outcomes]
        peak_infections = [outcome["peak_infections"] for outcome in self.outcomes]
        total_days = [outcome["total_days"] for outcome in self.outcomes]

        # Store aggregated results
        self.aggregated_data = {
            "total_infections": total_infections,
            "total_deaths": total_deaths,
            "peak_infections": peak_infections,
            "total_days": total_days,
        }

    def _aggregate_daily_stats(self) -> None:
        """Aggregate daily statistics across all runs and compute expected values."""
        # Determine the maximum number of days across all runs
        max_days = max(len(stats_collector.days) for stats_collector in self.stats)

        # Initialize daily aggregated data
        for day in range(max_days):
            day_data = {
                "susceptible": [],
                "infected": [],
                "infectious": [],
                "recovered": [],
                "dead": [],
                "vaccinated": [],
            }
            for stats_collector in self.stats:
                if day < len(stats_collector.days):
                    # Access counts from the respective lists
                    day_data["susceptible"].append(
                        stats_collector.susceptible_counts[day]
                    )
                    day_data["infected"].append(stats_collector.infected_counts[day])
                    day_data["infectious"].append(
                        stats_collector.infectious_counts[day]
                    )
                    day_data["recovered"].append(stats_collector.recovered_counts[day])
                    day_data["dead"].append(stats_collector.dead_counts[day])
                    day_data["vaccinated"].append(
                        stats_collector.vaccinated_counts[day]
                    )
            self.daily_aggregated_data.append(day_data)

        # Compute expected counts and standard deviations for each state
        for day_data in self.daily_aggregated_data:
            expected_counts = {}
            for state, counts in day_data.items():
                if counts:
                    expected_counts[state] = {
                        "mean": np.mean(counts),
                        "std": np.std(counts, ddof=1),
                    }
                else:
                    expected_counts[state] = {"mean": 0, "std": 0}
            self.daily_expected_counts.append(expected_counts)

    def plot_histograms(self) -> None:
        """Plot histograms of the aggregated metrics with statistical summaries."""
        if not self.aggregated_data:
            logger.warning("No statistics collected. Please run the simulations first.")

            return

        # Create a mapping of metric names to their data with readable labels
        metrics = {
            " ".join(key.title().split("_")): values
            for key, values in self.aggregated_data.items()
        }

        num_metrics = len(metrics)

        # Determine the number of rows and columns for subplots
        ncols = min(num_metrics, 3)  # Limit the number of columns to 3 for readability
        nrows = math.ceil(num_metrics / ncols)

        # Adjust the figure size accordingly
        plt.figure(figsize=(6 * ncols, 6 * nrows))

        for idx, (metric_name, values) in enumerate(metrics.items(), 1):
            plt.subplot(nrows, ncols, idx)
            plt.hist(values, bins=10, color="skyblue", edgecolor="black")
            plt.xlabel(metric_name)
            plt.ylabel("Frequency")
            plt.title(f"Histogram of {metric_name}")

            # Calculate statistical summaries
            mean_value = np.mean(values)
            median_value = np.median(values)
            std_dev = np.std(values, ddof=1)
            conf_interval = (
                1.96 * std_dev / np.sqrt(len(values))
            )  # 95% confidence interval

            # Add text annotations on the plot
            stats_text = (
                f"Mean: {mean_value:.2f}\n"
                f"Median: {median_value:.2f}\n"
                f"Std Dev: {std_dev:.2f}\n"
                f"95% CI: ±{conf_interval:.2f}"
            )
            # Position the text box in the upper right corner
            props = dict(boxstyle="round", facecolor="white", alpha=0.8)
            plt.text(
                0.95,
                0.95,
                stats_text,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                transform=plt.gca().transAxes,
                bbox=props,
            )

        plt.tight_layout()
        plt.show()

    def plot_state_histograms(self) -> None:
        """Plot the mean number of individuals in each state over time, each state in a separate subplot."""
        if not self.daily_expected_counts:
            logger.warning(
                "No daily expected counts found. Please run the simulations first."
            )
            return

        days = range(len(self.daily_expected_counts))
        states = [
            "susceptible",
            "infected",
            "infectious",
            "recovered",
            "dead",
            "vaccinated",
        ]

        num_states = len(states)
        ncols = min(num_states, 3)
        nrows = math.ceil(num_states / ncols)

        plt.figure(figsize=(6 * ncols, 4 * nrows))

        for idx, state in enumerate(states, 1):
            means = [day_data[state]["mean"] for day_data in self.daily_expected_counts]
            stds = [day_data[state]["std"] for day_data in self.daily_expected_counts]
            means = np.array(means)
            stds = np.array(stds)

            plt.subplot(nrows, ncols, idx)
            plt.plot(days, means, label=f"{state.capitalize()}", color="blue")
            plt.fill_between(days, means - stds, means + stds, color="blue", alpha=0.2)
            plt.xlabel("Day")
            plt.ylabel("Number of Individuals")
            plt.title(f"Expected Number of {state.capitalize()} Over Time")
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.show()
