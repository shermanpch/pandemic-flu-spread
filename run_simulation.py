import argparse
import itertools
import logging
import os
import shutil
import sys
from functools import partial

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
    ],
)

logger = logging.getLogger(__name__)


# Add src directory to sys.path
def add_src_to_path():
    """Add the src directory to the Python path."""
    current_dir = os.getcwd()
    src_path = os.path.join(current_dir, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


# Call the function before importing from src
add_src_to_path()

from src.batch_run import SimulationBatchRunner
from src.policies import (
    mask_policy,
    social_distancing_policy,
    supply_constrained_vaccination_policy,
)


def clear_or_create_directory(directory_path):
    """
    Clears all contents of the given directory if it exists.
    If it doesn't exist, creates the directory.
    """
    try:
        if os.path.exists(directory_path):
            logger.info(f"Directory '{directory_path}' exists. Clearing contents...")
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            logger.info(f"All contents of '{directory_path}' have been cleared.")
        else:
            logger.info(f"Directory '{directory_path}' does not exist. Creating it...")
            os.makedirs(directory_path)
            logger.info(f"Directory '{directory_path}' has been created.")
    except Exception as e:
        logger.error(f"Error processing directory '{directory_path}': {e}")


def run_simulation(debug=False):
    # Get the current working directory
    current_dir = os.getcwd()

    # Construct paths
    results_dir = os.path.join(current_dir, "results")

    # Clear or create results directory
    clear_or_create_directory(results_dir)

    infection_rates = [0.4, 0.5, 0.6, 0.7, 0.8]
    mask_start_days = [5, 10, 15]
    social_distancing_start_days = [5, 10, 15]
    replications = 100

    if debug:
        infection_rates = [0.4]
        mask_start_days = [5]
        social_distancing_start_days = [5]
        replications = 2

    # Initialize DataFrames for storing results
    histogram_results_df = pd.DataFrame()
    state_over_time_results_df = pd.DataFrame()
    loading_tracker = 0

    for (
        infection_rate,
        mask_start_day,
        social_distancing_start_day,
    ) in itertools.product(
        infection_rates,
        mask_start_days,
        social_distancing_start_days,
    ):
        logger.info(
            f"Running simulation with infection_rate={infection_rate}, "
            f"mask_start_day={mask_start_day}, "
            f"social_distancing_start_day={social_distancing_start_day}"
        )

        percent_complete = (
            loading_tracker
            / (
                len(infection_rates)
                * len(mask_start_days)
                * len(social_distancing_start_days)
            )
        ) * 100
        logger.info(f"Percent complete: {percent_complete:.2f}%")

        loading_tracker += 1

        MASK_POLICY = partial(
            mask_policy,
            start_day=mask_start_day,
            max_prob_mask=0.8,
            growth_rate=0.05,
        )

        SOCIAL_DIST_POLICY = partial(
            social_distancing_policy,
            start_day=social_distancing_start_day,
        )

        VAC_POLICY = partial(
            supply_constrained_vaccination_policy,
            initial_supply_rate=0.01,
            max_supply_rate=0.15,
            growth_type="logistic",
        )

        simulation_params = {
            "population_size": 10_000,
            "initial_infected": 0.01,
            "infection_rate": infection_rate,
            "incubation_period": 7,
            "infectious_period": 14,
            "base_contacts": 10,
            "social_distancing_rate": 0.3,
            "mortality_rate": 0.01,
            "mask_effectiveness": 0.5,
            "partial_vaccine_effectiveness": 0.5,
            "full_vaccine_effectiveness": 0,
            "mask_policy": MASK_POLICY,
            "dist_policy": SOCIAL_DIST_POLICY,
            "vac_policy": VAC_POLICY,
            "random_seed": 42,
        }

        batch_runner = SimulationBatchRunner(
            replications=replications,
            simulation_params=simulation_params,
        )

        batch_runner.run()

        hist_save_path = os.path.join(
            results_dir,
            f"ir{infection_rate}_ms{mask_start_day}_sds{social_distancing_start_day}_hist.png",
        )
        state_save_path = os.path.join(
            results_dir,
            f"ir{infection_rate}_ms{mask_start_day}_sds{social_distancing_start_day}_state.png",
        )

        title_hist = f"Statistical Summary\nInfection Rate: {infection_rate}\nMask Start Day: {mask_start_day}\nSocial Distancing Start Day: {social_distancing_start_day}"
        title_state = f"State Over Time\nInfection Rate: {infection_rate}\nMask Start Day: {mask_start_day}\nSocial Distancing Start Day: {social_distancing_start_day}"

        hist_df = batch_runner.plot_histograms(
            save_path=hist_save_path,
            print_graphs=False,
            title=title_hist,
        )

        hist_df = hist_df.assign(
            infection_rate=infection_rate,
            mask_start_day=mask_start_day,
            social_distancing_start_day=social_distancing_start_day,
        )

        histogram_results_df = pd.concat(
            [histogram_results_df, hist_df],
            ignore_index=True,
        )

        state_df = batch_runner.plot_state_over_time(
            save_path=state_save_path,
            print_graphs=False,
            title=title_state,
        )

        state_df = state_df.assign(
            infection_rate=infection_rate,
            mask_start_day=mask_start_day,
            social_distancing_start_day=social_distancing_start_day,
        )

        state_over_time_results_df = pd.concat(
            [state_over_time_results_df, state_df],
            ignore_index=True,
        )

    histogram_results_df.to_excel(
        os.path.join(results_dir, "histogram_results.xlsx"),
        index=False,
    )

    state_over_time_results_df.to_excel(
        os.path.join(results_dir, "state_over_time_results.xlsx"),
        index=False,
    )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Simulation script")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )

    args = parser.parse_args()

    # Set debug mode
    debug = args.debug
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode is enabled for simulation.")
    else:
        logger.info("Running simulation in normal mode.")

    # Simulate your main logic here
    run_simulation(debug=debug)


if __name__ == "__main__":
    main()
