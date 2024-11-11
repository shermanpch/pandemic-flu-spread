from typing import Callable

from person import Person
from simulation import Simulation


def mask_policy(simulation: Simulation, person: Person) -> None:
    """
    A policy function to determine if a person should wear a mask.

    Args:
        simulation (Simulation): The simulation instance.
        person (Person): The person to potentially apply the mask policy to.
    """
    # Example policy: After day 10, everyone starts wearing masks
    if simulation.day >= 10 and not person.masked:
        person.apply_mask()


def vulnerable_distancing_policy(simulation: Simulation, person: Person) -> None:
    """
    A policy function to determine if a person should practice social distancing.

    Args:
        simulation (Simulation): The simulation instance.
        person (Person): The person to potentially apply the social distancing policy to.
    """
    # Example policy: Vulnerable individuals start social distancing
    # For simplicity, let's assume individuals with ID < 20 are vulnerable
    if person.id < 20 and not person.social_distancing:
        person.start_social_distancing()


def general_distancing_policy(simulation: Simulation, person: Person) -> None:
    """
    A policy function where everyone practices social distancing after a certain day.

    Args:
        simulation (Simulation): The simulation instance.
        person (Person): The person to potentially apply the social distancing policy to.
    """
    # Example policy: After day 15, everyone practices social distancing
    if simulation.day >= 15 and not person.social_distancing:
        person.start_social_distancing()


def basic_vaccination_policy(simulation: Simulation) -> None:
    """
    A policy function to vaccinate a certain number of individuals each day.

    Args:
        simulation (Simulation): The simulation instance.
    """
    # Example policy: Vaccinate 10 susceptible individuals per day
    unvaccinated = [
        p
        for p in simulation.population
        if p.vaccination_doses == 0 and p.is_susceptible
    ]
    num_to_vaccinate = min(10, len(unvaccinated))
    to_vaccinate = simulation.random.sample(unvaccinated, k=num_to_vaccinate)
    for person in to_vaccinate:
        person.vaccinate()


def targeted_vaccination_policy(simulation: Simulation) -> None:
    """
    A policy function to prioritize vaccination for vulnerable individuals.

    Args:
        simulation (Simulation): The simulation instance.
    """
    # Example policy: Vaccinate vulnerable individuals first
    # Vulnerable individuals have ID < 20
    unvaccinated_vulnerable = [
        p
        for p in simulation.population
        if p.vaccination_doses == 0 and p.is_susceptible and p.id < 20
    ]
    if unvaccinated_vulnerable:
        num_to_vaccinate = min(10, len(unvaccinated_vulnerable))
        to_vaccinate = simulation.random.sample(
            unvaccinated_vulnerable, k=num_to_vaccinate
        )
    else:
        # If no vulnerable individuals left, vaccinate others
        unvaccinated = [
            p
            for p in simulation.population
            if p.vaccination_doses == 0 and p.is_susceptible
        ]
        num_to_vaccinate = min(10, len(unvaccinated))
        to_vaccinate = simulation.random.sample(unvaccinated, k=num_to_vaccinate)
    for person in to_vaccinate:
        person.vaccinate()


def compliance_mask_policy(
    simulation: Simulation,
    person: Person,
    compliance_rate: float,
) -> None:
    """
    A policy function where individuals wear masks based on a compliance rate.

    Args:
        simulation (Simulation): The simulation instance.
        person (Person): The person to potentially apply the mask policy to.
        compliance_rate (float): The probability that a person will wear a mask.
    """
    if not person.masked and simulation.random.random() < compliance_rate:
        person.apply_mask()


def time_based_vaccination_policy(
    simulation: Simulation,
    start_day: int,
    daily_vaccinations: int,
) -> None:
    """
    A policy function to start vaccinations after a certain day.

    Args:
        simulation (Simulation): The simulation instance.
        start_day (int): The day vaccinations begin.
        daily_vaccinations (int): Number of individuals to vaccinate each day.
    """
    if simulation.day >= start_day:
        unvaccinated = [
            p
            for p in simulation.population
            if p.vaccination_doses == 0 and p.is_susceptible
        ]
        num_to_vaccinate = min(daily_vaccinations, len(unvaccinated))
        to_vaccinate = simulation.random.sample(unvaccinated, k=num_to_vaccinate)
        for person in to_vaccinate:
            person.vaccinate()


def custom_mask_policy_factory(start_day: int) -> Callable[[Simulation, Person], None]:
    """
    Factory function to create a custom mask policy starting on a specific day.

    Args:
        start_day (int): The day the mask policy starts.

    Returns:
        Callable[[Simulation, Person], None]: The mask policy function.
    """

    def mask_policy(simulation: Simulation, person: Person) -> None:
        if simulation.day >= start_day and not person.masked:
            person.apply_mask()

    return mask_policy


def custom_distancing_policy_factory(
    start_day: int,
    compliance_rate: float,
) -> Callable[[Simulation, Person], None]:
    """
    Factory function to create a custom social distancing policy.

    Args:
        start_day (int): The day the distancing policy starts.
        compliance_rate (float): The probability that a person will practice social distancing.

    Returns:
        Callable[[Simulation, Person], None]: The distancing policy function.
    """

    def distancing_policy(simulation: Simulation, person: Person) -> None:
        if simulation.day >= start_day and not person.social_distancing:
            if simulation.random.random() < compliance_rate:
                person.start_social_distancing()

    return distancing_policy
