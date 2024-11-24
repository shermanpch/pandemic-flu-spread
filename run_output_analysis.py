import argparse
import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


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


def read_data(results_dir):
    """Reads the required data from Excel files in the results directory."""
    try:
        histogram_results = pd.read_excel(
            os.path.join(results_dir, "histogram_results.xlsx")
        )
        state_over_time_results = pd.read_excel(
            os.path.join(results_dir, "state_over_time_results.xlsx")
        )
        logger.info("Successfully loaded data from results directory.")
        return histogram_results, state_over_time_results
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        raise


def create_pivot_table(data, infection_rate):
    """Creates a pivot table for given infection rate."""
    df = data[data["infection_rate"] == infection_rate].copy()
    df["(mask_start_day, social_distancing_start_day)"] = df[
        ["mask_start_day", "social_distancing_start_day"]
    ].apply(tuple, axis=1)

    pivot_df = df.pivot_table(
        index=["(mask_start_day, social_distancing_start_day)"],
        columns="Metric",
        values="Mean",
    ).reset_index()

    pivot_df.columns.name = None

    return pivot_df


def plot_heatmap(data, infection_rate, metric_name, ax, vmin=None, vmax=None):
    """Plots a heatmap for a given metric and infection rate."""
    dataset_name = metric_name.replace("total_", "").replace("_", " ").capitalize()
    heatmap_data = data.pivot_table(
        index="social_distancing_start_day",
        columns="mask_start_day",
        values="Mean",
        aggfunc="mean",
    )
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=False,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        annot_kws={"size": 24},
    )

    ax.set_title(f"Mean {dataset_name}\nInfection Rate: {infection_rate}", fontsize=14)
    ax.set_xlabel("Mask Start Day", fontsize=14)
    ax.set_ylabel("Social Distancing Start Day", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)


def plot_line_graphs(data, infection_rates, title, save_path):
    """Plots line graphs for states across infection rates."""
    num_plots = len(infection_rates)
    ncols, nrows = 2, (num_plots + 1) // 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15, 5 * nrows),
        sharex=False,
        sharey=True,
    )

    axes = axes.flatten()
    colormap = ListedColormap(
        plt.cm.tab10(
            np.linspace(
                0,
                1,
                len(
                    data[
                        ["mask_start_day", "social_distancing_start_day"]
                    ].drop_duplicates()
                ),
            )
        )
    )

    for i, infection_rate in enumerate(infection_rates):
        if i < len(axes):
            ax = axes[i]
            subset = data[data["infection_rate"] == infection_rate]
            unique_combinations = subset[
                ["mask_start_day", "social_distancing_start_day"]
            ].drop_duplicates()

            for j, (mask_start_day, social_distancing_start_day) in enumerate(
                unique_combinations.values
            ):
                line_data = subset[
                    (subset["mask_start_day"] == mask_start_day)
                    & (
                        subset["social_distancing_start_day"]
                        == social_distancing_start_day
                    )
                ]
                ax.plot(
                    line_data["Day"],
                    line_data["Mean"],
                    label=f"mask: {mask_start_day}, social: {social_distancing_start_day}",
                    color=colormap(j),
                )

            ax.set_title(f"Infection Rate: {infection_rate}")
            ax.set_xlabel("Day")
            ax.set_ylabel("Number of Individuals")
            ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower right",
        ncol=2,
        title="(Mask Start Day, Social Distancing Start Day)",
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.savefig(save_path)
    logger.info(f"Line graph saved at {save_path}")
    plt.close(fig)


def process_and_plot_histogram_data(histogram_results, analysis_results_dir):
    """Processes and plots histogram results."""
    metrics = histogram_results["Metric"].unique()
    for metric in metrics:
        metric_data = histogram_results[histogram_results["Metric"] == metric]
        infection_rates = histogram_results["infection_rate"].unique()
        num_plots = len(infection_rates)

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))
        axes = axes.flatten()
        vmin, vmax = metric_data["Mean"].min(), metric_data["Mean"].max()

        for i, infection_rate in enumerate(infection_rates):
            if i < len(axes):
                data = metric_data[metric_data["infection_rate"] == infection_rate]
                plot_heatmap(data, infection_rate, metric, axes[i], vmin, vmax)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])

        fig.colorbar(
            plt.cm.ScalarMappable(
                cmap="YlGnBu", norm=plt.Normalize(vmin=vmin, vmax=vmax)
            ),
            cax=cbar_ax,
            orientation="horizontal",
        )

        cbar_ax.set_xlabel("Mean Infections", fontsize=14)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        fig.savefig(os.path.join(analysis_results_dir, f"{metric}_heatmaps.png"))
        logger.info(
            f"Heatmap for {metric} saved at {os.path.join(analysis_results_dir, f'{metric}_heatmaps.png')}"
        )
        plt.close(fig)


def run_output_analysis():
    # Paths
    current_dir = os.getcwd()
    results_dir = os.path.join(current_dir, "results")
    analysis_results_dir = os.path.join(current_dir, "analysis_results")

    # Prepare directories
    clear_or_create_directory(analysis_results_dir)

    # Load data
    histogram_results_df, state_over_time_results_df = read_data(results_dir)

    # Process and plot histogram data
    process_and_plot_histogram_data(histogram_results_df, analysis_results_dir)

    # Combine mask and social distancing start days into a tuple
    state_over_time_results_df["mask_start_day, social_distancing_start_day"] = (
        state_over_time_results_df[
            ["mask_start_day", "social_distancing_start_day"]
        ].apply(tuple, axis=1)
    )

    # Define state titles
    title_map = {
        "susceptible": " Expected Number of Susceptible Individuals as of Day X",
        "immune": "Expected Number of Immune Individuals as of Day X",
        "infected": "Expected Number of Infected Individuals as of Day X",
        "infectious": "Expected Number of Infectious Individuals as of Day X",
        "recovered": "Expected Number of Recovered Individuals as of Day X",
        "dead": "Expected Number of Dead Individuals as of Day X",
        "cumulative_vac_one": "Expected Number of Vaccinated Individuals (First Dose) as of Day X",
        "cumulative_vac_two": "Expected Number of Vaccinated Individuals (Second Dose) as of Day X",
        "cumulative_masked": "Expected Number of Masked Individuals as of Day X",
        "vac_one": "Expected Number of New Daily Vaccinated Individuals (First Dose)",
        "vac_two": "Expected Number of New Daily Vaccinated Individuals (Second Dose)",
        "masking": "Expected Number of New Daily Masked Individuals",
        "vaccine_supply": "Number of New Daily Vaccine Doses",
    }

    # Plot line graphs for states
    infection_rates = histogram_results_df["infection_rate"].unique()
    for state in state_over_time_results_df["State"].unique():
        state_data = state_over_time_results_df[
            state_over_time_results_df["State"] == state
        ]
        plot_line_graphs(
            state_data,
            infection_rates,
            title_map.get(state, state),
            os.path.join(analysis_results_dir, f"{state}.png"),
        )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analysis script")
    args = parser.parse_args()

    logger.info("Running analysis...")
    # Simulate analysis logic here
    run_output_analysis()


if __name__ == "__main__":
    main()
