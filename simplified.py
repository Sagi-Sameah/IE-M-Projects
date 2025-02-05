import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.ndimage import gaussian_filter1d  # For smoothing

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
    def __init__(self, probability, projects):
        """
        Args:
            probability (float): Probability of supporting a project.
            projects (list[Project]): List of all available projects.
        """
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
        sum = 0
        for project in selectedP:
            if self.supportedP.__contains__(project):
                sum+=1
        return sum/len(self.supportedP)

############################ Dynamic Generation Functions #########################
def generate_projects(num_projects, cost_range=(2000, 10000)):
    """
    Generate a list of projects with random costs.
    """
    projects = []
    for i in range(num_projects):
        name = f"Project_{i + 1}"
        cost = random.randint(*cost_range)
        projects.append(Project(name, cost))
    return projects

def generate_voters(num_voters, probability, projects):
    """
    Generate a list of voters with specified probability and project list.
    """
    voters = []
    support_matrix = []
    for _ in range(num_voters):
        voter = Voter(probability, projects)
        voters.append(voter)
        # Create binary support row for the matrix
        support_row = [1 if project in voter.supportedP else 0 for project in projects]
        support_matrix.append(support_row)
    return voters, np.array(support_matrix)

##################### Aggregation Method #######################
def greedy_selection(scores, projects, budget):
    """
    Select projects using the greedy approval algorithm.
    Args:
        scores (list[float]): Project scores.
        projects (list[Project]): List of projects.
        budget (float): Budget constraint.
    Returns:
        list[Project]: Selected projects.
    """
    sorted_projects = sorted(zip(projects, scores), key=lambda x: -x[1])  # Sort by score descending
    # print("Project Order for Greedy:")
    # for idx, (project, score) in enumerate(sorted_projects):  # Unpack the tuple (project, score)
        # print(f"Index {idx}: {project.name}, score: {score:.2f}")

    selected_projects = []
    total_cost = 0
    for project, score in sorted_projects:
        if total_cost + project.cost <= budget:
            selected_projects.append(project)
            total_cost += project.cost
    print(f"total cost: {total_cost}")
    return selected_projects

##################### Simulation Functions #######################
def run_simulation(num_projects, num_voters, budget, probability, cost_range, num_replications, warm_up_period, fixed_seed):
    """
    Run the participatory budgeting simulation.
    """
    set_random_seed(fixed_seed)  # Set seed for reproducibility or variability

    print("Starting simulation...")
    print(f"Generating {num_projects} projects and {num_voters} voters...")

    projects = generate_projects(num_projects, cost_range)
    print("Projects generated:")
    for project in projects:
        print(f"  {project}")
    print(f"Budget: {budget}")

    turnout_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    average_satisfaction_scores = []  # For average satisfaction

    for turnout in turnout_percentages:
        print(f"\nSimulating for turnout: {turnout}%")
        replications_satisfaction = []  # Store satisfaction for each replication
        final_support_matrix = None
        final_selected_projects = None

        for rep in range(num_replications + warm_up_period):
            voters, support_matrix = generate_voters(num_voters, probability, projects)
            num_voters_to_select = int(len(voters) * turnout / 100)
            participating_voters = random.sample(voters, num_voters_to_select)

            # Restrict the support matrix to participating voters only
            participating_indices = [voters.index(voter) for voter in participating_voters]
            restricted_support_matrix = support_matrix[participating_indices]

            # Calculate column tally as scores for the restricted support matrix
            column_tally = restricted_support_matrix.sum(axis=0)
            print(column_tally)

            # Perform project selection using the greedy algorithm
            selected_projects = greedy_selection(column_tally, projects, budget)

            if rep >= warm_up_period:  # Exclude warm-up replications
                total_satisfaction = sum(voter.calculate_satisfaction(selected_projects) for voter in participating_voters)
                print(f"Total Satisfaction (Turnout {turnout}%, Replication {rep}): {total_satisfaction}")

                average_satisfaction = total_satisfaction / num_voters
                replications_satisfaction.append(average_satisfaction)

                # Save the restricted support matrix and selected projects for the last replication
                final_support_matrix = restricted_support_matrix
                final_selected_projects = selected_projects

        # Average satisfaction over replications
        avg_satisfaction = sum(replications_satisfaction) / len(replications_satisfaction)
        average_satisfaction_scores.append(avg_satisfaction)

        # Generate the binary selectedP vector
        selectedP = [1 if project in final_selected_projects else 0 for project in projects]

        # Print the final support matrix, column tally, and selectedP vector for this turnout
        print(f"\nFinal Support Matrix (Turnout {turnout}%):")
        for voter_idx, row in enumerate(final_support_matrix):
            voter_satisfaction = participating_voters[voter_idx].calculate_satisfaction(final_selected_projects)
            print(f"Voter {voter_idx + 1}: {row} | Satisfaction: {voter_satisfaction:.2f}")
        print(f"\nColumn Tally: {column_tally}")
        print("Project Order for Tallying:")
        for idx, project in enumerate(projects):
            print(f"Index {idx}: {project.name}")
        print(f"Selected Projects (selectedP): {selectedP}")
        print("Selected Projects Validation:")
        for idx, val in enumerate(selectedP):
            if val == 1:
                print(f"Project Selected: {projects[idx].name}")
        print(f"Turnout {turnout}% - Average Satisfaction: {avg_satisfaction:.3f}")

    print("\nSimulation completed.")
    return turnout_percentages, average_satisfaction_scores

##################### Plotting Results ###########################
def plot_satisfaction(turnout_percentages, satisfaction_scores):
    """
    Plot the satisfaction vs. turnout graph.
    """
    smoothed_scores = gaussian_filter1d(satisfaction_scores, sigma=1.5)

    plt.figure(figsize=(10, 6))
    plt.plot(turnout_percentages, satisfaction_scores, marker='o', linestyle='-', color='b', label='Original Satisfaction')
    plt.plot(turnout_percentages, smoothed_scores, marker='', linestyle='--', color='r', label='Smoothed Satisfaction')
    plt.title("Average Satisfaction vs. Turnout Percentage (Greedy Algorithm)")
    plt.xlabel("Turnout Percentage (%)")
    plt.ylabel("Average Satisfaction Score")
    plt.legend()
    plt.grid(True)
    plt.show()

##################### Run Simulation ############################
turnout_percentages, average_satisfaction_scores = run_simulation(
    num_projects=10,
    num_voters=100,
    budget=9000,
    probability=0.5,
    cost_range=(2000, 4000),
    num_replications=100,
    warm_up_period=0,
    fixed_seed=False  # Use random seed for variability
)

# Plot the results
plot_satisfaction(turnout_percentages, average_satisfaction_scores)
