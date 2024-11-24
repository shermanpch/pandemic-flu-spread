import argparse
import logging
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def run_script(script_path, debug=False):
    """
    Runs a Python script as a subprocess.

    Args:
        script_path (str): The path to the script to execute.
        debug (bool): Whether to enable debug mode for the script.
    """
    try:
        cmd = ["python", script_path]
        if debug:
            cmd.append("--debug")
            logger.debug(f"Running {script_path} in debug mode.")
        else:
            logger.info(f"Running {script_path}.")

        subprocess.run(cmd, check=True)
        logger.info(f"Successfully ran {script_path}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Script {script_path} failed with exit code {e.returncode}.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while running {script_path}: {e}")
        raise


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run simulation and analysis scripts.")
    parser.add_argument(
        "--run-simulation-only",
        action="store_true",
        help="Run only the simulation script.",
    )

    parser.add_argument(
        "--run-analysis-only",
        action="store_true",
        help="Run only the analysis script.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for the simulation script.",
    )

    args = parser.parse_args()

    # Paths to the scripts
    simulation_script = "run_simulation.py"
    analysis_script = "run_output_analysis.py"

    try:
        # Decide which scripts to run based on flags
        if args.run_simulation_only:
            logger.info("Running simulation script only.")
            run_script(simulation_script, debug=args.debug)
        elif args.run_analysis_only:
            logger.info("Running analysis script only.")
            run_script(analysis_script)
        else:
            logger.info("Running both simulation and analysis scripts.")
            run_script(simulation_script, debug=args.debug)
            run_script(analysis_script)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
