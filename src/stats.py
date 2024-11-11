from typing import Dict, List, Set

import matplotlib.pyplot as plt

from person import Person


class StatisticsCollector:
    """
    Collects and manages statistics during the simulation run.

    Attributes:
        days (List[int]): List of day numbers.
        susceptible_counts (List[int]): Number of susceptible individuals each day.
        infected_counts (List[int]): Number of infected (not yet infectious) individuals each day.
        infectious_counts (List[int]): Number of infectious individuals each day.
        recovered_counts (List[int]): Number of recovered individuals each day.
        dead_counts (List[int]): Number of dead individuals each day.
        vaccinated_counts (List[int]): Number of vaccinated individuals each day.
        total_infected_set (Set[int]): Set of unique individuals who have been infected.
        peak_infections (int): Maximum number of infectious individuals at any point.
    """

    def __init__(self):
        """Initialize a new StatisticsCollector instance."""
        self.days: List[int] = []
        self.susceptible_counts: List[int] = []
        self.infected_counts: List[int] = []
        self.infectious_counts: List[int] = []
        self.recovered_counts: List[int] = []
        self.dead_counts: List[int] = []
        self.vaccinated_counts: List[int] = []
        self.total_infected_set: Set[int] = set()
        self.peak_infections: int = 0

    def record_day(self, day: int, population: List[Person]) -> None:
        """
        Record the statistics for the current day.

        Args:
            day (int): The current day number.
            population (List[Person]): The list of all individuals in the simulation.
        """
        # Count the number of individuals in each state
        susceptible = sum(1 for person in population if person.is_susceptible)
        infected = sum(1 for person in population if person.is_infected)
        infectious = sum(1 for person in population if person.is_infectious)
        recovered = sum(1 for person in population if person.is_recovered)
        dead = sum(1 for person in population if person.is_dead)
        vaccinated = sum(1 for person in population if person.vaccination_doses > 0)

        # Append the counts to the lists
        self.days.append(day)
        self.susceptible_counts.append(susceptible)
        self.infected_counts.append(infected)
        self.infectious_counts.append(infectious)
        self.recovered_counts.append(recovered)
        self.dead_counts.append(dead)
        self.vaccinated_counts.append(vaccinated)

        # Update the set of total infected individuals
        for person in population:
            if (
                person.is_infected
                or person.is_infectious
                or person.is_recovered
                or person.is_dead
            ):
                self.total_infected_set.add(person.id)

        # Update peak infections
        current_infections = infectious
        if current_infections > self.peak_infections:
            self.peak_infections = current_infections

    def get_data(self) -> Dict[str, List[int]]:
        """
        Return the collected data as a dictionary.

        Returns:
            Dict[str, List[int]]: A dictionary containing the lists of daily counts.
        """
        return {
            "days": self.days,
            "susceptible": self.susceptible_counts,
            "infected": self.infected_counts,
            "infectious": self.infectious_counts,
            "recovered": self.recovered_counts,
            "dead": self.dead_counts,
            "vaccinated": self.vaccinated_counts,
        }

    def get_final_data(self) -> Dict[str, int]:
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

    def plot(self) -> None:
        """
        Plot the collected data over time using matplotlib.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.days, self.susceptible_counts, label="Susceptible")
        plt.plot(self.days, self.infected_counts, label="Infected")
        plt.plot(self.days, self.infectious_counts, label="Infectious")
        plt.plot(self.days, self.recovered_counts, label="Recovered")
        plt.plot(self.days, self.dead_counts, label="Dead")
        plt.plot(self.days, self.vaccinated_counts, label="Vaccinated")
        plt.xlabel("Day")
        plt.ylabel("Number of Individuals")
        plt.title("Disease Progression Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()
