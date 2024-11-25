from typing import Dict, List, Set

import matplotlib.pyplot as plt
from person import Person


class StatisticsCollector:
    """
    Collects and manages statistics during the simulation run.
    """

    def __init__(self):
        """Initialize a new StatisticsCollector instance."""
        self.days: List[int] = []
        self.susceptible_counts: List[int] = []
        self.immune_counts: List[int] = []
        self.infected_counts: List[int] = []
        self.infectious_counts: List[int] = []
        self.recovered_counts: List[int] = []
        self.dead_counts: List[int] = []
        self.cumulative_vac_one_counts: List[int] = []
        self.cumulative_vac_two_counts: List[int] = []
        self.cumulative_masked_counts: List[int] = []
        self.vac_one_counts: List[int] = []
        self.vac_two_counts: List[int] = []
        self.masking_counts: List[int] = []
        self.vaccine_supply_counts: List[int] = []
        self.total_infected_set: Set[int] = set()
        self.peak_infections: int = 0

    def record_day(
        self,
        day: int,
        population: List[Person],
        vaccine_supply: int,
    ) -> None:
        """
        Record the statistics for the current day.

        Args:
            day (int): The current day number.
            population (List[Person]): The list of all individuals in the simulation.
        """
        # Initialize counts
        daily_counts = {
            "susceptible": 0,
            "immune": 0,
            "infected": 0,
            "infectious": 0,
            "recovered": 0,
            "dead": 0,
            "cumulative_vac_one": 0,
            "cumulative_vac_two": 0,
            "cumulative_masked": 0,
            "vaccine_supply": vaccine_supply,
        }

        # Count individuals in each health state
        for person in population:
            state = person.get_health_state()

            if state in daily_counts:
                daily_counts[state] += 1

            # Check if the individual is vaccinated
            if person.vaccination_doses == 1:
                daily_counts["cumulative_vac_one"] += 1

            elif person.vaccination_doses == 2:
                daily_counts["cumulative_vac_two"] += 1

            if person.masked:
                daily_counts["cumulative_masked"] += 1

            # Check if the individual should be added to the total infected set
            if state in ["infected", "infectious", "recovered", "dead"]:
                self.total_infected_set.add(person.id)

        # Update the class attributes with counts
        self.days.append(day)
        self.susceptible_counts.append(daily_counts["susceptible"])
        self.immune_counts.append(daily_counts["immune"])
        self.infected_counts.append(daily_counts["infected"])
        self.infectious_counts.append(daily_counts["infectious"])
        self.recovered_counts.append(daily_counts["recovered"])
        self.dead_counts.append(daily_counts["dead"])
        self.cumulative_vac_one_counts.append(daily_counts["cumulative_vac_one"])
        self.cumulative_vac_two_counts.append(daily_counts["cumulative_vac_two"])
        self.cumulative_masked_counts.append(daily_counts["cumulative_masked"])
        self.vaccine_supply_counts.append(daily_counts["vaccine_supply"])

        # Calculate daily vaccinated and masked counts
        if day == 0:
            # First day: daily count equals cumulative count
            self.vac_one_counts.append(daily_counts["cumulative_vac_one"])
            self.vac_two_counts.append(daily_counts["cumulative_vac_two"])
            self.masking_counts.append(daily_counts["cumulative_masked"])
        else:
            # Subsequent days: calculate daily difference
            self.vac_one_counts.append(
                max(
                    0,
                    daily_counts["cumulative_vac_one"]
                    - self.cumulative_vac_one_counts[-2],
                )
            )
            self.vac_two_counts.append(
                daily_counts["cumulative_vac_two"] - self.cumulative_vac_two_counts[-2]
            )
            self.masking_counts.append(
                daily_counts["cumulative_masked"] - self.cumulative_masked_counts[-2]
            )

        # Update peak infections
        current_infections = daily_counts["infectious"]
        self.peak_infections = max(self.peak_infections, current_infections)

    def get_outcome(self) -> Dict[str, int]:
        """
        Extract key metrics at the end of the simulation.

        Returns:
            Dict[str, int]: A dictionary containing total infections, total deaths, and peak infections.
        """
        total_infections = len(self.total_infected_set)
        total_deaths = self.dead_counts[-1] if self.dead_counts else 0
        peak_infections = self.peak_infections
        total_days = len(self.days)

        return {
            "total_infections": total_infections,
            "total_deaths": total_deaths,
            "peak_infections": peak_infections,
            "total_days": total_days,
        }

    def plot(self, metrics_to_plot: List[str] = None) -> None:
        """
        Plot the collected data over time using matplotlib.

        Args:
            metrics_to_plot (List[str], optional): List of specific metrics to plot (e.g., ["susceptible", "infected"]).
        """
        plt.figure(figsize=(10, 6))

        metric_map = {
            "susceptible": self.susceptible_counts,
            "immune": self.immune_counts,
            "infected": self.infected_counts,
            "infectious": self.infectious_counts,
            "recovered": self.recovered_counts,
            "dead": self.dead_counts,
            "cumulative_vac_one": self.cumulative_vac_one_counts,
            "cumulative_vac_two": self.cumulative_vac_two_counts,
            "cumulative_masked": self.cumulative_masked_counts,
            "vac_one": self.vac_one_counts,
            "vac_two": self.vac_two_counts,
            "masking": self.masking_counts,
            "vaccine_supply": self.vaccine_supply_counts,
        }

        title_map = {
            "susceptible": "Number of Susceptible Individuals as of Day X",
            "immune": "Number of Immune Individuals as of Day X",
            "infected": "Number of Infected Individuals as of Day X",
            "infectious": "Number of Infectious Individuals as of Day X",
            "recovered": "Number of Recovered Individuals as of Day X",
            "dead": "Number of Dead Individuals as of Day X",
            "cumulative_vac_one": "Number of Vaccinated Individuals (First Dose) as of Day X",
            "cumulative_vac_two": "Number of Vaccinated Individuals (Second Dose) as of Day X",
            "cumulative_masked": "Number of Masked Individuals as of Day X",
            "vac_one": "Number of New Daily Vaccinated Individuals (First Dose)",
            "vac_two": "Number of New Daily Vaccinated Individuals (Second Dose)",
            "masking": "Number of New Daily Masked Individuals",
            "vaccine_supply": "Number of New Daily Vaccine Doses",
        }

        if metrics_to_plot is None:
            metrics_to_plot = metric_map.keys()

        for metric in metrics_to_plot:
            if metric in metric_map:
                plt.plot(
                    self.days,
                    metric_map[metric],
                    label=title_map[metric],
                )

        plt.xlabel("Day")
        plt.ylabel("Number of Individuals")
        plt.title("Disease Progression Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()
