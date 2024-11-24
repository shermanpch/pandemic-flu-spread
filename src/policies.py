import logging
import math
import sys

from person import Person
from simulation import Simulation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
    ],
)

logger = logging.getLogger(__name__)


# Mask Policy (Module Level)
def mask_policy(
    simulation: Simulation,
    person: Person,
    start_day: int,
    max_prob_mask: float,
    growth_rate: float,
) -> None:
    """
    A policy function to determine if a person should wear a mask, with probability increasing over time.

    Args:
        simulation (Simulation): The simulation instance.
        person (Person): The person to potentially apply the mask policy to.
        start_day (int): The day on which the mask policy starts.
        max_prob_mask (float): The maximum probability of a person wearing a mask.
        growth_rate (float, optional): The rate at which the probability increases over time. Defaults to 0.1.
    """
    # Check if the current day is at or past the start day for masking policy
    if simulation.day >= start_day and not person.masked:

        # Calculate days since start of mask policy
        days_since_start = simulation.day - start_day

        # Calculate the probability using a logistic function
        prob_mask = max_prob_mask * (1 - math.exp(-growth_rate * days_since_start))

        # Apply mask with the calculated probability
        if simulation.random.random() < prob_mask:
            person.apply_mask()
            logger.debug(
                f"Person {person.id} has started wearing a mask on day {simulation.day} with probability {prob_mask:.2f}."
            )


# Social Distancing Policy (Module Level)
def social_distancing_policy(
    simulation: Simulation,
    person: Person,
    start_day: int,
) -> None:
    """
    A policy function where everyone practices social distancing after a certain day.

    Args:
        simulation (Simulation): The simulation instance.
        person (Person): The person to potentially apply the social distancing policy to.
        start_day (int): The day on which the social distancing policy starts.
    """
    if simulation.day >= start_day and not person.social_distancing:
        person.start_social_distancing()
        logger.debug(
            f"Person {person.id} started social distancing on day {simulation.day}."
        )


# Vaccination Policy (Module Level)
def supply_constrained_vaccination_policy(
    simulation: Simulation,
    initial_supply_rate: float,
    max_supply_rate: float,
    growth_type: str = "logistic",
) -> None:
    """
    A policy function to simulate a vaccination strategy with limited vaccine supply.

    Args:
        simulation (Simulation): The simulation instance.
        initial_supply_rate (float): Starting supply as a fraction of the population.
        max_supply_rate (float): Maximum daily supply as a fraction of the population.
        daily_increase_rate (float): Rate at which supply increases each day as a fraction of the population.
        growth_type (str): Type of growth ('linear', 'exponential', 'logistic').
    """
    population_size = len(simulation.population)
    available_doses = get_daily_vaccine_supply(
        day=simulation.day,
        population_size=population_size,
        initial_supply_rate=initial_supply_rate,
        max_supply_rate=max_supply_rate,
        growth_type=growth_type,
    )

    if available_doses > 0:
        unvaccinated = [
            p
            for p in simulation.population
            if p.is_susceptible and p.vaccination_doses < p.MAX_VACCINATION_DOSES
        ]
        num_to_vaccinate = min(available_doses, len(unvaccinated))
        to_vaccinate = simulation.random.sample(unvaccinated, k=num_to_vaccinate)

        for person in to_vaccinate:
            person.vaccinate()
            logger.debug(
                f"Person {person.id} vaccinated on day {simulation.day} with {available_doses} available doses."
            )

    return available_doses


def get_daily_vaccine_supply(
    day: int,
    population_size: int,
    initial_supply_rate: float = 0.01,
    max_supply_rate: float = 0.15,
    growth_type: str = "logistic",
):
    """
    Calculate the vaccine supply for a given day based on a specified growth model.

    Args:
        day (int): The current day in the simulation.
        population_size (int): The total population size of the simulation.
        initial_supply_rate (float): Starting supply as a fraction of the population.
        max_supply_rate (float): Maximum daily supply as a fraction of the population.
        growth_type (str): Type of growth ('linear', 'exponential', 'logistic').

    Returns:
        int: The number of doses available for the given day.
    """
    carrying_capacity = max_supply_rate * population_size
    midpoint = 15  # Arbitrary midpoint for logistic growth

    if growth_type == "linear":
        # Linearly increase supply each day up to the maximum supply rate
        daily_supply = min(
            initial_supply_rate * population_size + (day * 0.005 * population_size),
            carrying_capacity,
        )
    elif growth_type == "exponential":
        # Exponential increase to represent early rapid scaling of production
        daily_supply = min(
            initial_supply_rate * population_size * (1.05**day),
            carrying_capacity,
        )
    elif growth_type == "logistic":
        # Logistic growth: slow start, rapid increase, and then leveling off
        growth_rate = 0.2  # Controls the steepness of the curve
        daily_supply = carrying_capacity / (
            1 + math.exp(-growth_rate * (day - midpoint))
        )
    else:
        raise ValueError(
            f"Unknown growth_type '{growth_type}'. Use 'linear', 'exponential', or 'logistic'."
        )

    return int(daily_supply)
