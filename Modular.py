import numpy as np
import random
import matplotlib.pyplot as plt


# ############################## Random Seed Toggle ############################## #
def set_random_seed(fixed_seed=True, seed_value=42):
    """
    Set random seeds for reproducibility or variability.

    Args:
        fixed_seed (bool): If True, uses a fixed seed for reproducibility.
        seed_value (int): Value of the fixed seed (default: 42).
    """
    if fixed_seed:
        random.seed(seed_value)
        np.random.seed(seed_value)
    else:
        random.seed()  # System-based seed
        np.random.seed(None)


# ############################## Part 1: Generating Population ############################## #
class Entity:
    """Abstract base class for entities (Voters, Projects)."""

    def __init__(self, position=None, probability=None, space_dim=(0, 1)):
        self.position = position if position is not None else np.random.uniform(*space_dim, 2)
        self.probability = probability  # Only relevant for voters


class Voter(Entity):
    """Represents a voter with a probability of supporting projects."""

    def __init__(self, probability, position=None, space_dim=(0, 1), env_type="simple"):
        super().__init__(position, probability, space_dim)
        self.env_type = env_type

    def supports(self, project):
        """Determines if the voter supports a project."""
        if self.env_type == "euclidean":
            distance = np.linalg.norm(self.position - project.position)
            return random.random() < self.probability * (1 / (1 + distance))
        return random.random() < self.probability


class VoterWithCoins(Voter):
    """Voter with coin-based support allocation."""

    def __init__(self, probability, projects, coins=1):
        super().__init__(probability)
        self.coins = coins
        self.projects = projects

    def distribute_coins(self):
        if not self.projects:
            return []
        allocation = [random.uniform(0, 1) for _ in self.projects]
        return [coin / sum(allocation) for coin in allocation]


class Project(Entity):
    """Represents a project with a cost attribute."""

    def __init__(self, cost, position=None, space_dim=(0, 1)):
        super().__init__(position, None, space_dim)
        self.cost = cost


def generate_population(num_voters, num_projects, model_type="simple", group_params=None, space_dim=(0, 1)):
    """
    Generates voters and projects based on specified parameters.
    """
    voters = []
    projects = [Project(cost=random.randint(1000, 5000)) for _ in range(num_projects)]

    if group_params:
        for group in group_params["groups"]:
            for _ in range(group["size"]):
                voters.append(
                    Voter(probability=group["probability"], env_type="simple", space_dim=space_dim)
                )
    else:
        for _ in range(num_voters):
            if model_type == "coin_unified":
                voters.append(VoterWithCoins(probability=random.uniform(0, 1), projects=projects))
            elif model_type == "coin_divergent":
                voters.append(
                    VoterWithCoins(
                        probability=random.uniform(0, 1), projects=projects, coins=random.uniform(0.5, 1.5)
                    )
                )
            else:  # Default to "simple"
                voters.append(Voter(probability=random.uniform(0, 1), env_type="simple", space_dim=space_dim))

    return voters, projects


# ############################## Part 2: Aggregation Methods ############################## #
def greedy_approval(project_scores, budget):
    """Implements a greedy approval algorithm."""
    sorted_projects = sorted(project_scores.items(), key=lambda x: -x[1])  # Sort by score
    selected_projects = []
    total_cost = 0

    for project, score in sorted_projects:
        if total_cost + project.cost <= budget:
            selected_projects.append(project)
            total_cost += project.cost

    return selected_projects


# ############################## Part 3: Satisfaction Metrics ############################## #
def cumulative_satisfaction(voters, selected_projects):
    """Calculate cumulative satisfaction."""
    return sum(
        sum([voter.supports(project) for project in selected_projects]) for voter in voters
    )


# ############################## Part 4: Simulation ############################## #
def simulate(
        num_voters, num_projects, budget, aggregation_method, satisfaction_metric,
        model_type="simple", group_params=None, space_dim=(0, 1), turnout_levels=None,
        num_replications=10, warmup=5, fixed_seed=True
):
    """
    Main simulation function.
    """
    set_random_seed(fixed_seed)  # Set the random seed for reproducibility or variability
    voters, projects = generate_population(num_voters, num_projects, model_type, group_params, space_dim)
    project_scores = {project: random.uniform(0, 1) for project in projects}  # Placeholder scoring

    if turnout_levels is None:
        turnout_levels = np.linspace(0.1, 1, 10)

    results = {"turnout": [], "satisfaction": []}

    for turnout in turnout_levels:
        satisfaction_scores = []
        for _ in range(num_replications + warmup):
            participating_voters = random.sample(voters, int(len(voters) * turnout))
            selected_projects = aggregation_method(project_scores, budget)
            if _ >= warmup:  # Ignore warmup iterations
                satisfaction_scores.append(satisfaction_metric(participating_voters, selected_projects))
        avg_satisfaction = np.mean(satisfaction_scores)
        results["turnout"].append(turnout)
        results["satisfaction"].append(avg_satisfaction)

    return results


# ############################## Part 5: Plotting ############################## #
def plot_results(results):
    plt.figure(figsize=(10, 6))
    plt.plot(results["turnout"], results["satisfaction"], marker="o", label="Satisfaction")
    plt.xlabel("Turnout Levels")
    plt.ylabel("Satisfaction Metric")
    plt.title("Satisfaction vs Turnout")
    plt.legend()
    plt.grid()
    plt.show()


# ############################## Test Case ############################## #
if __name__ == "__main__":
    results = simulate(
        num_voters=100,
        num_projects=10,
        budget=15000,
        aggregation_method=greedy_approval,
        satisfaction_metric=cumulative_satisfaction,
        model_type="simple",
        group_params={
            "groups": [
                {"probability": 0.8, "size": 50},
                {"probability": 0.5, "size": 50},
            ]
        },
        turnout_levels=np.linspace(0.1, 1, 10),
        num_replications=30,
        warmup=5,
        fixed_seed=False,  # Set to False for variability
    )
    plot_results(results)
