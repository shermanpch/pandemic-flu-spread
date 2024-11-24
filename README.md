
# Pandemic Flu Spread Simulation

This project models the spread of a pandemic flu under various policy interventions, enabling analysis of their effectiveness through simulations and visualized outputs.

## Project Overview

The project includes the following components:

1. **Simulation**: Models the spread of the flu based on configurable parameters and policy interventions.
2. **Analysis**: Processes simulation results to generate insights through visualizations and statistical summaries.

### Key Features

- **Flexible Policy Configuration**: Includes mask-wearing, social distancing, and vaccination policies.
- **Scalable Simulations**: Runs multiple replications for statistical reliability.
- **Automated Analysis**: Generates visual heatmaps, line graphs, and detailed reports.
- **Extensibility**: Modular codebase for customizing policies and analysis.

## Project Structure

- **Scripts**:
  - `run_simulation.py`: Executes the simulation with various configurations and policies.
  - `run_output_analysis.py`: Analyzes simulation results and produces visualizations.
  - `main.py`: Orchestrates the workflow, allowing users to run simulations, analyses, or both.
- **Source Code**: `src/` contains modules for simulation logic and policy definitions.
- **Results**: Simulation outputs are stored in `results/` and analysis outputs in `analysis_results/`.

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/pandemic-flu-spread.git
   cd pandemic-flu-spread
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### General Workflow

1. Run the **simulation** to generate outputs:
   ```bash
   python main.py --run-simulation-only
   ```

2. Analyze the outputs to generate reports and visualizations:
   ```bash
   python main.py --run-analysis-only
   ```

3. Or run both steps together:
   ```bash
   python main.py
   ```

### Debug Mode

Debug mode runs the simulation with reduced complexity for faster testing:
```bash
python main.py --debug
```

## Configuration

Key parameters for the simulation can be modified directly in `run_simulation.py`, including:

- **Infection rates**: Explore different transmission scenarios.
- **Policy start days**: Customize when mask-wearing or social distancing begins.
- **Population and replication settings**: Adjust the scale and reliability of simulations.

## Outputs

1. **Simulation Results** (`results/`):
   - Statistical summaries and state-over-time data in `.xlsx` files.
   - Visual outputs such as histograms and line graphs.

2. **Analysis Results** (`analysis_results/`):
   - Heatmaps highlighting policy effectiveness.
   - Line graphs showing the dynamics of the outbreak over time.

## Logging

Logs provide real-time updates on the simulation and analysis process, configurable via the `logging.basicConfig` settings in each script.
