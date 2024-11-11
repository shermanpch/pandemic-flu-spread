import random
from enum import Enum


class HealthState(Enum):
    SUSCEPTIBLE = "susceptible"
    INFECTED = "infected"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"
    DEAD = "dead"


class Person:
    """Represents an individual in the simulation with health status and behaviors."""

    MAX_VACCINATION_DOSES: int = 2  # Maximum number of vaccine doses

    def __init__(self, id: int):
        """
        Initialize a new Person instance.

        Args:
            id (int): A unique identifier for the person.
        """
        self.id: int = id
        self.state: HealthState = HealthState.SUSCEPTIBLE
        self.days_infected: int = 0  # Counter for infection duration
        self.vaccination_doses: int = 0  # Number of vaccine doses received
        self.masked: bool = False  # Whether the person is wearing a mask
        self.social_distancing: bool = False  # Whether the person is social distancing

    @property
    def is_susceptible(self) -> bool:
        """Check if the person is susceptible."""
        return self.state == HealthState.SUSCEPTIBLE

    @property
    def is_infected(self) -> bool:
        """Check if the person is infected (but not yet infectious)."""
        return self.state == HealthState.INFECTED

    @property
    def is_infectious(self) -> bool:
        """Check if the person is infectious."""
        return self.state == HealthState.INFECTIOUS

    @property
    def is_recovered(self) -> bool:
        """Check if the person has recovered."""
        return self.state == HealthState.RECOVERED

    @property
    def is_dead(self) -> bool:
        """Check if the person is dead."""
        return self.state == HealthState.DEAD

    def infected(self) -> None:
        """Transition the person to the 'infected' state if susceptible."""
        if self.is_susceptible:
            self.state = HealthState.INFECTED
            self.days_infected = 0  # Reset the counter

    def become_infectious(self) -> None:
        """Transition the person from 'infected' to 'infectious' state."""
        if self.is_infected:
            self.state = HealthState.INFECTIOUS

    def recover_or_die(self, mortality_rate: float, rng: random.Random) -> None:
        """
        Determine whether the person recovers or dies based on the mortality rate.

        Args:
            mortality_rate (float): Probability of dying from the disease (0 <= mortality_rate <= 1).
            rng (random.Random): Random number generator instance for reproducibility.
        """
        if self.is_infectious:
            if rng.random() < mortality_rate:
                self.state = HealthState.DEAD
            else:
                self.state = HealthState.RECOVERED

    def vaccinate(self, doses: int = 1) -> None:
        """
        Administer vaccine doses to the person.

        Args:
            doses (int): Number of doses to administer (default is 1).
        """
        self.vaccination_doses += doses
        self.vaccination_doses = min(self.vaccination_doses, self.MAX_VACCINATION_DOSES)

    def apply_mask(self) -> None:
        """Set the person's masked status to True."""
        self.masked = True

    def remove_mask(self) -> None:
        """Set the person's masked status to False."""
        self.masked = False

    def start_social_distancing(self) -> None:
        """Set the person's social distancing status to True."""
        self.social_distancing = True

    def stop_social_distancing(self) -> None:
        """Set the person's social distancing status to False."""
        self.social_distancing = False

    def update_health_status(
        self,
        incubation_period: int,
        infectious_period: int,
        mortality_rate: float,
        rng: random.Random,
    ) -> None:
        """
        Update the person's health status based on incubation and infectious periods.

        Args:
            incubation_period (int): Number of days before becoming infectious.
            infectious_period (int): Number of days the person remains infectious.
            mortality_rate (float): Probability of dying from the disease (0 <= mortality_rate <= 1).
            rng (random.Random): Random number generator instance for reproducibility.
        """
        if self.is_infected:
            self.days_infected += 1
            if self.days_infected >= incubation_period:
                self.become_infectious()

        elif self.is_infectious:
            self.days_infected += 1
            if self.days_infected >= incubation_period + infectious_period:
                self.recover_or_die(mortality_rate, rng)
