import random
from typing import Callable, Generator, List, Optional

import simpy

from person import Person
from stats import StatisticsCollector


class Simulation:
    """
    Manages the simulation of disease spread among a population.

    Attributes:
        env (simpy.Environment): The simulation environment.
        population (List[Person]): List of Person instances representing the population.
        infection_rate (float): Base infection probability per contact.
        incubation_period (int): Number of days before an infected person becomes infectious.
        infectious_period (int): Number of days an infectious person remains infectious.
        base_contacts (int): Number of contacts per day for non-social distancing individuals.
        social_distancing_rate (float): Rate to reduce contacts for social distancing individuals.
        mortality_rate (float): Probability of dying from the disease (0 <= mortality_rate <= 1).
        mask_policy (Optional[Callable[['Simulation', Person], None]]): Function implementing the mask-wearing policy.
        dist_policy (Optional[Callable[['Simulation', Person], None]]): Function implementing the social distancing policy.
        vac_policy (Optional[Callable[['Simulation'], None]]): Function implementing the vaccination strategy.
        statistics_collector (Optional[StatisticsCollector]): Collects statistics during the simulation.
        random_seed (Optional[int]): Seed for the random number generator.
        day (int): The current day in the simulation.
    """

    def __init__(
        self,
        env: simpy.Environment,
        population_size: int,
        initial_infected: int,
        infection_rate: float,
        incubation_period: int,
        infectious_period: int,
        base_contacts: int,
        social_distancing_rate: float,
        mortality_rate: float = 0.01,
        mask_effectiveness: float = 0.5,
        partial_vaccine_effectiveness: float = 0.7,
        full_vaccine_effectiveness: float = 0.0,
        mask_policy: Optional[Callable[["Simulation", Person], None]] = None,
        dist_policy: Optional[Callable[["Simulation", Person], None]] = None,
        vac_policy: Optional[Callable[["Simulation"], None]] = None,
        statistics_collector: Optional[StatisticsCollector] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize a new Simulation instance.

        Args:
            env (simpy.Environment): The simulation environment that manages time and events.
            population_size (int): Total number of individuals in the simulation.
            initial_infected (Union[int, float]): Number or fraction of individuals initially infected.
                If an integer, it represents the absolute number of individuals infected at the start.
                If a float between 0 and 1, it specifies the fraction of the population to be initially infected.
            infection_rate (float): Base probability of infection per contact (0 <= infection_rate <= 1).
            incubation_period (int): Number of days from infection to becoming infectious.
            infectious_period (int): Number of days an individual remains infectious.
            base_contacts (int): Average number of contacts per day for individuals who are not social distancing.
            social_distancing_rate (float): Multiplier to reduce daily contacts for social distancing individuals
                (0 <= social_distancing_rate <= 1). A rate of 0.5 means half as many contacts.
            mortality_rate (float): Probability of dying from the disease for infected individuals
                (0 <= mortality_rate <= 1).
            mask_effectiveness (float): Multiplier to reduce infection probability due to mask usage.
                If a person wears a mask, the base infection rate is multiplied by this factor, reducing it.
                For example, with a `mask_effectiveness` of 0.5, the infection rate is halved when a mask is worn.
            partial_vaccine_effectiveness (float): Multiplier to reduce infection probability for individuals with one dose of the vaccine.
                The infection rate is multiplied by this factor for partially vaccinated individuals.
                For example, a value of 0.7 reduces the infection rate by 30%.
            full_vaccine_effectiveness (float): Multiplier to reduce infection probability for individuals with full vaccination (two or more doses).
                The infection rate is multiplied by this factor for fully vaccinated individuals.
                For example, a value of 0.0 effectively makes fully vaccinated individuals immune to infection.
            mask_policy (Optional[Callable]): Function that implements the mask-wearing policy for individuals.
                This function is called on each person daily to determine if they should wear a mask.
            dist_policy (Optional[Callable]): Function that implements the social distancing policy for individuals.
                This function is called on each person daily to determine their social distancing behavior.
            vac_policy (Optional[Callable]): Function that implements the vaccination strategy for the simulation.
                This function is called daily to determine which individuals should receive vaccines.
            statistics_collector (Optional[StatisticsCollector]): Collects and manages statistical data during the simulation.
                This is used to record information on the states of individuals and the progress of the disease.
            random_seed (Optional[int]): Seed for the random number generator to ensure reproducibility of the simulation.
        """
        # Validation checks
        if population_size <= 0:
            raise ValueError("Population size must be positive.")
        # Validate initial_infected
        if isinstance(initial_infected, int):
            if not (0 <= initial_infected <= population_size):
                raise ValueError(
                    "Initial infected must be between 0 and population size."
                )
            num_initial_infected = initial_infected
        elif isinstance(initial_infected, float):
            if not (0 <= initial_infected <= 1):
                raise ValueError(
                    "If initial_infected is a float, it must be between 0 and 1."
                )
            num_initial_infected = int(population_size * initial_infected)
        else:
            raise TypeError("initial_infected must be an int or a float.")
        if not 0 <= infection_rate <= 1:
            raise ValueError("Infection rate must be between 0 and 1.")
        if incubation_period <= 0:
            raise ValueError("Incubation period must be positive.")
        if infectious_period <= 0:
            raise ValueError("Infectious period must be positive.")
        if base_contacts < 0:
            raise ValueError("Base contacts must be non-negative.")
        if not 0 <= social_distancing_rate <= 1:
            raise ValueError("Social distancing rate must be between 0 and 1.")
        if not 0 <= mortality_rate <= 1:
            raise ValueError("Mortality rate must be between 0 and 1.")

        self.env: simpy.Environment = env
        self.population: List[Person] = [Person(id=i) for i in range(population_size)]
        self.infection_rate: float = infection_rate
        self.incubation_period: int = incubation_period
        self.infectious_period: int = infectious_period
        self.base_contacts: int = base_contacts
        self.social_distancing_rate: float = social_distancing_rate
        self.mortality_rate: float = mortality_rate
        self.mask_effectiveness: float = mask_effectiveness
        self.partial_vaccine_effectiveness: float = partial_vaccine_effectiveness
        self.full_vaccine_effectiveness: float = full_vaccine_effectiveness
        self.mask_policy: Optional[Callable[["Simulation", Person], None]] = mask_policy
        self.dist_policy: Optional[Callable[["Simulation", Person], None]] = dist_policy
        self.vac_policy: Optional[Callable[["Simulation"], None]] = vac_policy
        self.statistics_collector: Optional[StatisticsCollector] = statistics_collector
        self.day: int = 0

        # Initialize random number generator
        self.random_seed: Optional[int] = random_seed
        self.random: random.Random = random.Random(self.random_seed)

        # Infect some initial individuals
        initial_infected_ids = self.random.sample(
            range(population_size),
            num_initial_infected,
        )
        for idx in initial_infected_ids:
            self.population[idx].infected()

        # Start the daily simulation process
        self.env.process(self.run())

    def run(self) -> Generator[simpy.events.Event, None, None]:
        """Run the simulation, advancing one day at a time."""
        while True:
            vaccine_supply = self.simulate_day()
            if self.statistics_collector is not None:
                self.statistics_collector.record_day(
                    self.day,
                    self.population,
                    vaccine_supply,
                )

            # Check if there are any infected or infectious individuals left
            active_infections = any(
                person.is_infected or person.is_infectious for person in self.population
            )
            if not active_infections:
                # No more infections; end the simulation
                break

            self.day += 1
            yield self.env.timeout(1)  # Proceed to the next day

    def simulate_day(self) -> None:
        """
        Simulate the events of a single day.

        This includes updating health statuses, applying policies (mask, distancing, vaccination),
        and simulating interactions between individuals.
        """
        self.update_health_statuses()
        vaccine_supply = self.apply_policies()
        self.simulate_interactions()
        return vaccine_supply

    def update_health_statuses(self) -> None:
        """Update the health status of each person."""
        for person in self.population:
            person.update_health_status(
                self.incubation_period,
                self.infectious_period,
                self.mortality_rate,
                self.random,  # Pass the Random instance
            )

    def apply_policies(self) -> None:
        """Apply preventive measures and vaccination policies."""
        for person in self.population:
            if self.mask_policy:
                self.mask_policy(self, person)
            if self.dist_policy:
                self.dist_policy(self, person)
        if self.vac_policy:
            available_doses = self.vac_policy(self)
        return available_doses

    def simulate_interactions(self) -> None:
        """Simulate interactions between individuals and possible disease transmission."""
        # Use sets for efficient membership testing and updates
        susceptible_population = [p for p in self.population if p.is_susceptible]
        infectious_population = [p for p in self.population if p.is_infectious]

        for infectious_person in infectious_population:
            # Determine number of contacts based on social distancing
            if infectious_person.social_distancing:
                num_contacts = int(self.base_contacts * self.social_distancing_rate)
            else:
                num_contacts = self.base_contacts

            # Ensure at least one contact if num_contacts is less than 1
            num_contacts = max(1, num_contacts)
            num_contacts = min(num_contacts, len(susceptible_population))
            if num_contacts == 0:
                continue  # No susceptible individuals left to infect

            # Select contacts randomly from available susceptible individuals
            contacts = self.random.sample(susceptible_population, k=num_contacts)
            for susceptible_person in contacts:
                infection_chance = self.calculate_infection_chance(
                    infectious_person,
                    susceptible_person,
                )
                if self.random.random() < infection_chance:
                    susceptible_person.infected()

    def apply_mask_effect(self, base_rate: float, person: Person) -> float:
        """
        Apply mask effectiveness to reduce the base infection rate.

        Masks reduce the infection rate by a fixed percentage based on their effectiveness.
        The infection rate is multiplied by `mask_effectiveness` for masked individuals,
        thereby lowering the likelihood of transmission.

        Args:
            base_rate (float): The initial infection rate before adjustments.
            person (Person): The individual for whom the mask effect is being applied.

        Returns:
            float: The adjusted infection rate considering mask effectiveness. If the person
                is masked, the infection rate will be `base_rate * mask_effectiveness`.
                Otherwise, it will remain as `base_rate`.
        """
        if person.masked:
            return base_rate * self.mask_effectiveness
        return base_rate

    def apply_vaccine_effect(self, base_rate: float, person: Person) -> float:
        """
        Apply vaccination effectiveness to reduce the base infection rate.

        The infection rate is multiplied by either `partial_vaccine_effectiveness` or
        `full_vaccine_effectiveness` based on the vaccination status of the individual.
        This effectively reduces the infection rate, thereby decreasing the chance of transmission.

        Args:
            base_rate (float): The initial infection rate before adjustments.
            person (Person): The individual for whom the vaccination effect is being applied.

        Returns:
            float: The adjusted infection rate considering vaccination status. If the person
                has received one dose, the infection rate is `base_rate * partial_vaccine_effectiveness`.
                If the person has received two or more doses, the rate is `base_rate * full_vaccine_effectiveness`.
                If unvaccinated, the rate remains as `base_rate`.
        """
        if person.vaccination_doses == 1:
            return base_rate * self.partial_vaccine_effectiveness
        elif person.vaccination_doses >= 2:
            return base_rate * self.full_vaccine_effectiveness
        return base_rate

    def calculate_infection_chance(
        self,
        infectious_person: Person,
        susceptible_person: Person,
    ) -> float:
        """
        Calculate the probability of infection during an interaction.

        This method calculates the infection chance based on base infection rate,
        mask effectiveness, and vaccination effectiveness of the individuals involved.

        Args:
            infectious_person (Person): The infectious individual.
            susceptible_person (Person): The susceptible individual.

        Returns:
            float: The adjusted infection probability, ensuring that it does not exceed 1.0.
        """
        base_rate = self.infection_rate
        base_rate = self.apply_mask_effect(base_rate, infectious_person)
        base_rate = self.apply_mask_effect(base_rate, susceptible_person)
        base_rate = self.apply_vaccine_effect(base_rate, susceptible_person)
        return min(base_rate, 1.0)
