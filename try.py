import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.ndimage import gaussian_filter1d  # For smoothing

############################ PRINT TOGGLE ############################
PRINT_ENABLED = True  # Set to False to disable all prints

def log(*args, **kwargs):
    """Helper function to control console printing."""
    if PRINT_ENABLED:
        print(*args, **kwargs)

############################ Fixed Seed Toggle ############################
def set_random_seed(fixed_seed=True, seed_value=42):
    """Set random seeds for reproducibility."""
    if fixed_seed:
        random.seed(seed_value)
        np.random.seed(seed_value)
    else:
        random.seed()  # Use system-based random seed
        np.random.seed(None)

############################ Projects #########################
class Project:
    def __init__(self, name, cost):
        self.name = name
        self.cost = cost

    def __repr__(self):
        return f"Project(name={self.name}, cost={self.cost})"

class Voter:
    def __init__(self, voter_id, probability, projects):
        """
        Args:
            voter_id (int): Unique identifier for the voter.
            probability (float): Probability of supporting a project.
            projects (list[Project]): List of all available projects.
        """
        self.voter_id = voter_id
        self.probability = probability
        self.supportedP = [project for project in projects if random.random() < probability]  # Supported projects

    def calculate_satisfaction(self, selectedP):
        """
        Calculate satisfaction for the voter.
        Args:
            selectedP (list[Project]): List of selected projects.
        Returns:
            float: Satisfaction score.
        """
        if len(self.supportedP) == 0:
            return 0  # Avoid division by zero
        sum1 = 0
        for project in selectedP:
            if project in self.supportedP:
                sum1 += 1
        return sum1 / len(self.supportedP)

############################ Dynamic Generation Functions #########################
def generate_projects(num_projects, cost_range=(2000, 10000)):
    """Generate a list of projects with random costs."""
    projects = []
    for i in range(num_projects):
        name = f"Project_{i + 1}"
        cost = random.randint(*cost_range)
        projects.append(Project(name, cost))
    return projects

def generate_voters(num_voters, probability, projects):
    """Generate a list of voters with specified probability and project list."""
    voters = []
    for voter_id in range(num_voters):
        voter = Voter(voter_id, probability, projects)
        voters.append(voter)
    return voters

##################### Aggregation Method #######################
def greedy_selection(scores, projects, budget):
    """Select projects using the greedy approval algorithm."""
    sorted_projects = sorted(zip(projects, scores), key=lambda x: -x[1])  # Sort by score descending
    selected_projects = []
    total_cost = 0
    for project, score in sorted_projects:
        if total_cost + project.cost <= budget:
            selected_projects.append(project)
            total_cost += project.cost
    return selected_projects

##################### Simulation Functions #######################
def run_simulation(num_projects, num_voters, budget, probability, cost_range, num_replications, warm_up_period, fixed_seed=True):
    """Run the participatory budgeting simulation with stratified sampling and bootstrapping."""
    set_random_seed(fixed_seed)  # Set seed for reproducibility or variability

    log("Starting simulation...")
    log(f"Generating {num_projects} projects and {num_voters} voters...\n")

    projects = generate_projects(num_projects, cost_range)
    log("Projects generated:")
    for project in projects:
        log(f"  {project}")
    log(f"\nBudget: {budget}\n")

    # Generate the full voter population once (100% turnout)
    voters_100 = generate_voters(num_voters, probability, projects)

    # Store the initial votes of 100% turnout voters for tracking
    voter_preferences = {voter.voter_id: set(voter.supportedP) for voter in voters_100}

    log("\nSupported Projects for Each 100% Voter:")
    for voter_index, voter in enumerate(voters_100[:100]):  # Limit output for brevity
        log(f"  Voter {voter.voter_id}: Supported Projects: {[project.name for project in voter.supportedP]}")
    log("=" * 40)

    turnout_percentages = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    satisfaction_100_voters = []  # Satisfaction of 100% voters at each X% turnout
    previous_satisfaction = None  # Track satisfaction at previous turnout

    for turnout in turnout_percentages:
        log(f"\n==== Simulating for {turnout}% Turnout ====")
        replication_results = []  # Store results for bootstrapping

        for rep in range(num_replications):
            # Stratified Sampling
            num_voters_to_select = int(len(voters_100) * turnout / 100)
            participating_voters = random.sample(voters_100, num_voters_to_select)

            # Track whether sampled voters retain the same vote
            for voter in participating_voters:
                original_vote = voter_preferences[voter.voter_id]
                current_vote = set(voter.supportedP)
                if original_vote != current_vote:
                    log(f"  WARNING: Voter {voter.voter_id} changed vote! Original: {[p.name for p in original_vote]}, New: {[p.name for p in current_vote]}")

            # Create the support matrix for the participating voters
            support_matrix = np.array([
                [1 if project in voter.supportedP else 0 for project in projects]
                for voter in participating_voters
            ])

            # Print the support matrix with voter IDs
            log(f"\nSupport Matrix for Replication {rep + 1}, Turnout {turnout}%:")
            log("Voter ID | " + " | ".join([project.name for project in projects]))
            for i, voter in enumerate(participating_voters):
                log(f"{voter.voter_id:8} | " + " | ".join(map(str, support_matrix[i])))

            # Calculate column tally as scores for the restricted support matrix
            column_tally = support_matrix.sum(axis=0)

            # Perform project selection using the greedy algorithm
            selected_projects = greedy_selection(column_tally, projects, budget)

            # Calculate satisfaction for 100% turnout voters
            satisfaction_scores = [
                voter.calculate_satisfaction(selected_projects) for voter in voters_100
            ]
            total_satisfaction = sum(satisfaction_scores)
            num_voters = len(satisfaction_scores)
            avg_satisfaction = total_satisfaction / num_voters

            # Print the sum, length, and average satisfaction
            log(f"  Total Satisfaction: {total_satisfaction:.3f}, Number of Voters: {num_voters}, Average Satisfaction: {avg_satisfaction:.3f}")

            replication_results.append(avg_satisfaction)

        # Average satisfaction over replications (bootstrapping)
        avg_satisfaction_100_voters = sum(replication_results) / len(replication_results)
        satisfaction_100_voters.append(avg_satisfaction_100_voters)

        log(f"  - Satisfaction of 100% Voters with {turnout}% Results: {avg_satisfaction_100_voters:.3f}")
        log("-" * 40)

        # If satisfaction decreases, print all relevant data
        if previous_satisfaction is not None and avg_satisfaction_100_voters < previous_satisfaction:
            log(f"\n⚠️  ALERT: Satisfaction decreased at {turnout}% turnout! Previous: {previous_satisfaction:.3f}, Now: {avg_satisfaction_100_voters:.3f}")

            # Log selected projects
            log("\nSelected Projects:")
            for project in selected_projects:
                log(f"  {project.name} (Cost: {project.cost})")

            # Log satisfaction scores
            log("\nVoter Satisfaction Scores:")
            for voter_index, satisfaction in enumerate(satisfaction_scores):
                log(f"  Voter {voter_index}: Satisfaction {satisfaction:.3f}")

            # Log support matrix
            log("\nSupport Matrix for This Turnout:")
            log("Voter ID | " + " | ".join([project.name for project in projects]))
            for i, voter in enumerate(participating_voters):
                log(f"{voter.voter_id:8} | " + " | ".join(map(str, support_matrix[i])))

            log("=" * 80)

        previous_satisfaction = avg_satisfaction_100_voters  # Update previous satisfaction

    log("\nSimulation completed.")
    return turnout_percentages, satisfaction_100_voters

##################### Plot Satisfaction ###########################
def plot_satisfaction_comparison(turnout_percentages, satisfaction_100_voters):
    """
    Plot satisfaction for 100% turnout voters.
    """
    smoothed_satisfaction = gaussian_filter1d(satisfaction_100_voters, sigma=1.5)

    plt.figure(figsize=(10, 6))
    # Plot average satisfaction for 100% turnout voters
    plt.plot(turnout_percentages, satisfaction_100_voters, marker='o', label='100% Voter Satisfaction (Actual)')
    # Plot smoothed satisfaction
    plt.plot(turnout_percentages, smoothed_satisfaction, linestyle='--', label='Smoothed Satisfaction')
    plt.title("100% Voter Satisfaction vs. Turnout Percentage")
    plt.xlabel("Turnout Percentage (%)")
    plt.ylabel("Average Satisfaction")
    plt.legend()
    plt.grid(True)
    plt.show()

##################### Run Simulation with Adjustments ############################
turnout_percentages, satisfaction_100_voters = run_simulation(
    num_projects=7,
    num_voters=100,
    budget=10000,
    probability=0.7,
    cost_range=(2000, 5000),
    num_replications=100,
    warm_up_period=0,
    fixed_seed=False
)

# Plot Satisfaction Comparison
plot_satisfaction_comparison(turnout_percentages, satisfaction_100_voters)
