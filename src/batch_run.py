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
    outcome = stats_collector.get_outcome()

    return {
        "outcome": outcome,
        "stats_collector": stats_collector,
        "run_number": run_number,
    }


class SimulationBatchRunner:
    """
    Runs multiple simulations and aggregates the results for statistical analysis.

    Attributes:
        replications (int): Number of simulation runs to perform.
        simulation_params (dict): Parameters for the simulations.
        stats (List[StatisticsCollector]): Statistics from each simulation run.
        outcomes (List[Dict[str, int]]): Key metrics from each run.
        aggregated_data (Dict[str, List[float]]): Aggregated results across all runs.
        daily_aggregated_data (List[Dict[str, List[float]]]): Aggregated daily data.
        daily_expected_counts (List[Dict[str, Dict[str, float]]]): Expected counts per day.
    """

    def __init__(self, replications: int, simulation_params: dict):
        """
        Initialize the SimulationBatchRunner.

        Args:
            replications (int): Number of simulation runs to perform.
            simulation_params (dict): Parameters for the simulations.
        """
        self.replications = replications
        self.simulation_params = simulation_params
        self.stats: List[StatisticsCollector] = []
        self.outcomes: List[Dict[str, int]] = []
        self.aggregated_data: Dict[str, List[float]] = {}
        self.daily_aggregated_data: List[Dict[str, List[float]]] = []
        self.daily_expected_counts: List[Dict[str, Dict[str, float]]] = []

    def run(self) -> None:
        """Run all simulations and aggregate the results using multiprocessing."""
        logger.info("Starting batch run of simulations.")
        # Prepare arguments for each simulation run
        simulation_args = [
            (self.simulation_params, run_number)
            for run_number in range(self.replications)
        ]

        with multiprocessing.Pool() as pool:
            try:
                results = pool.map(run_single_simulation, simulation_args)
            except Exception as e:
                logger.error(f"Error occurred during simulation runs: {e}")

        logger.info("All simulations completed. Processing results...")

        for result in results:
            outcome = result["outcome"]
            stats_collector = result["stats_collector"]
            run_number = result["run_number"]

            self.stats.append(stats_collector)
            self.outcomes.append(outcome)
            logger.debug(f"Simulation {run_number + 1}/{self.replications} completed.")

        self._aggregate_results()
        self._aggregate_daily_stats()
        logger.info("Results aggregation completed.")

    def _aggregate_results(self) -> None:
        """Aggregate key metrics across all runs by processing individual statistics."""
        self._aggregate_metric("total_infections")
        self._aggregate_metric("total_deaths")
        self._aggregate_metric("peak_infections")
        self._aggregate_metric("total_days")

    def _aggregate_metric(self, metric: str) -> None:
        """Helper method to aggregate a specific metric."""
        metric_values = [outcome[metric] for outcome in self.outcomes]
        self.aggregated_data[metric] = metric_values

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
                "masked": [],  # Track masked individuals for the new feature
            }
            for sc in self.stats:
                if day < len(sc.days):
                    # Access counts from the respective lists
                    day_data["susceptible"].append(sc.susceptible_counts[day])
                    day_data["infected"].append(sc.infected_counts[day])
                    day_data["infectious"].append(sc.infectious_counts[day])
                    day_data["recovered"].append(sc.recovered_counts[day])
                    day_data["dead"].append(sc.dead_counts[day])
                    day_data["vaccinated"].append(sc.vaccinated_counts[day])
                    day_data["masked"].append(sc.masked_counts[day])
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

    def plot_histograms(
        self,
        metrics_to_plot: List[str] = None,
        ncols: int = 3,
        color: str = "skyblue",
    ) -> None:
        """
        Plot histograms of the aggregated metrics with statistical summaries.

        Args:
            metrics_to_plot (List[str], optional): The metrics to plot. If None, plot all available metrics.
            ncols (int): Number of columns in the plot grid.
            color (str): Color of the histogram bars.
        """
        metrics = metrics_to_plot if metrics_to_plot else self.aggregated_data.keys()
        num_metrics = len(metrics)

        nrows = math.ceil(num_metrics / ncols)
        plt.figure(figsize=(6 * ncols, 6 * nrows))

        for idx, metric in enumerate(metrics, 1):
            values = self.aggregated_data.get(metric, [])
            if not values:
                continue  # Skip if no data is available

            plt.subplot(nrows, ncols, idx)
            plt.hist(values, bins=10, color=color, edgecolor="black")
            plt.xlabel(metric.replace("_", " ").title())
            plt.ylabel("Frequency")
            plt.title(f"Histogram of {metric.replace('_', ' ').title()}")

            mean_value = np.mean(values)
            median_value = np.median(values)
            std_dev = np.std(values, ddof=1)
            conf_interval = 1.96 * std_dev / np.sqrt(len(values))

            stats_text = (
                f"Mean: {mean_value:.2f}\n"
                f"Median: {median_value:.2f}\n"
                f"Std Dev: {std_dev:.2f}\n"
                f"95% CI: ±{conf_interval:.2f}"
            )
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

    def plot_state_over_time(self) -> None:
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

    def plot_policy_over_time(self) -> None:
        """Plot the expected number of masked and vaccinated individuals over time."""
        if not self.daily_expected_counts:
            logger.warning(
                "No daily expected counts found. Please run the simulations first."
            )
            return

        days = range(len(self.daily_expected_counts))

        plt.figure(figsize=(12, 6))

        # Plot for the expected number of masked individuals over time
        plt.subplot(1, 2, 1)
        masked_means = [
            day_data.get("masked", {"mean": 0})["mean"]
            for day_data in self.daily_expected_counts
        ]
        masked_stds = [
            day_data.get("masked", {"std": 0})["std"]
            for day_data in self.daily_expected_counts
        ]

        plt.plot(days, masked_means, label="Masked", color="green")
        plt.fill_between(
            days,
            np.array(masked_means) - np.array(masked_stds),
            np.array(masked_means) + np.array(masked_stds),
            color="green",
            alpha=0.2,
        )
        plt.xlabel("Day")
        plt.ylabel("Number of Individuals")
        plt.title("Expected Number of Masked Individuals Over Time")
        plt.grid(True)
        plt.legend()

        # Plot for the expected number of vaccinated individuals over time
        plt.subplot(1, 2, 2)
        vaccinated_means = [
            day_data["vaccinated"]["mean"] for day_data in self.daily_expected_counts
        ]
        vaccinated_stds = [
            day_data["vaccinated"]["std"] for day_data in self.daily_expected_counts
        ]

        plt.plot(days, vaccinated_means, label="Vaccinated", color="blue")
        plt.fill_between(
            days,
            np.array(vaccinated_means) - np.array(vaccinated_stds),
            np.array(vaccinated_means) + np.array(vaccinated_stds),
            color="blue",
            alpha=0.2,
        )
        plt.xlabel("Day")
        plt.ylabel("Count")
        plt.title("Expected Supply of Vaccines Over Time")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
