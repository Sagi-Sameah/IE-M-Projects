import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from itertools import combinations

############################ Projects #########################
class Project:
    def __init__(self, name, cost, position):
        self.name = name
        self.cost = cost
        self.position = position  # Each project has a position in space (for proximity)

    def __repr__(self):
        return f"Project(name={self.name}, cost={self.cost}, position={self.position})"

############################ Voters and Voting #########################
class Vote:
    def __init__(self, project_votes):
        self.project_votes = project_votes

    def __repr__(self):
        return f"Vote({self.project_votes})"

class Voter:
    def __init__(self, probability, projects, position=None):
        self.probability = probability
        self.projects = projects
        self.position = position if position is not None else np.random.rand(2)
        self.vote = None

    def get_vote(self):
        """Retrieve the stored vote for consistent simulations."""
        return self.vote

    def cast_vote(self, budget):
        """Cast a vote based on proximity, influenced by a probability and budget constraint."""
        selected_projects = []
        total_cost = 0
        for project in self.projects:
            distance = np.linalg.norm(self.position - project.position)
            vote_probability = self.probability * (1 / (1 + distance))
            if random.random() < vote_probability and (total_cost + project.cost <= budget):
                selected_projects.append(1)
                total_cost += project.cost
            else:
                selected_projects.append(0)
        self.vote = Vote(selected_projects)

class VoterWithCoins(Voter):
    def __init__(self, probability, projects, coins=1):
        super().__init__(probability, projects)
        self.coins = coins

    def distribute_coins(self):
        """Distribute voting coins across selected projects."""
        if self.vote is None:
            return [0] * len(self.projects)
        total_votes = sum(self.vote.project_votes)
        if total_votes == 0:
            return [0] * len(self.projects)
        return [(self.coins / total_votes) * vote for vote in self.vote.project_votes]

############################ Ballot Box #########################
class BallotBoxWithCoins:
    def __init__(self):
        self.ballots = []

    def add_ballot(self, vote_distribution):
        """Add a voter's coin distribution to the ballot box."""
        self.ballots.append(vote_distribution)

    def tally_coin_votes(self):
        """Aggregate the coin distributions across all ballots to get project scores."""
        if not self.ballots:
            return [0] * len(self.ballots[0])  # Handle empty case
        project_scores = [0] * len(self.ballots[0])
        for ballot in self.ballots:
            for i in range(len(project_scores)):
                project_scores[i] += ballot[i]
        return project_scores

############################ Dynamic Generation #########################
def generate_projects(num_projects, cost_range=(2000, 10000), space_dim=(0, 1)):
    projects = []
    for i in range(num_projects):
        name = f"Project_{i+1}"
        cost = random.randint(*cost_range)
        position = np.random.uniform(space_dim[0], space_dim[1], 2)
        projects.append(Project(name, cost, position))
    return projects

def generate_voters(num_voters, probability, projects, space_dim=(0, 1)):
    voters = []
    for i in range(num_voters):
        position = np.random.uniform(space_dim[0], space_dim[1], 2)
        voters.append(VoterWithCoins(probability, projects, coins=1))
    return voters

##################### Greedy Algorithm #######################
def get_applicable_combinations(projects, budget):
    valid_combinations = []
    for r in range(1, len(projects) + 1):
        for comb in combinations(projects, r):
            total_cost = sum(p.cost for p in comb)
            if total_cost <= budget:
                valid_combinations.append(comb)
    return valid_combinations

def greedy_selection(valid_combinations, scores, projects):
    """Select the project combination with the highest satisfaction score within budget."""
    project_to_score = {project.name: score for project, score in zip(projects, scores)}
    selected_projects = sorted(valid_combinations, key=lambda comb: -sum(project_to_score[p.name] for p in comb))
    return selected_projects[0]

##################### Simulation Functions #######################
def simulate_turnout(voters, turnout_percentage):
    num_voters_to_select = int(len(voters) * turnout_percentage / 100)
    return random.sample(voters, num_voters_to_select)

def calculate_cumulative_satisfaction(voters, selected_projects, projects):
    """Calculate cumulative satisfaction for the given voters and selected projects."""
    cumulative_satisfaction = 0
    selected_project_names = {project.name for project in selected_projects}

    for voter in voters:
        vote = voter.get_vote()
        # print(vote)
        if vote:
            satisfaction = sum([1 for i, selected in enumerate(vote.project_votes)
                                if selected and projects[i].name in selected_project_names])
            cumulative_satisfaction += satisfaction

    return cumulative_satisfaction

################################# Main Simulation Program ############################
def run_simulation(num_projects=5, num_voters=100, budget=11000, probability=0.7, cost_range=(2000, 10000), space_dim=(0, 1), num_replications=30, warm_up_period=5):
    projects = generate_projects(num_projects, cost_range, space_dim)
    valid_combinations = get_applicable_combinations(projects, budget)
    turnout_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cumulative_greedy_satisfaction = []

    for turnout in turnout_percentages:
        total_cumulative_satisfaction = 0
        final_selected_projects = []  # To store the final selected projects
        final_project_scores = None  # To store the final project scores

        for rep in range(num_replications + warm_up_period):
            random.seed(rep)  # Set a unique seed for each replication
            np.random.seed(rep)  # Set numpy's random seed for consistency

            voters = generate_voters(num_voters, probability, projects)
            turnout_voters = simulate_turnout(voters, turnout)

            if rep >= warm_up_period:
                ballot_box = BallotBoxWithCoins()
                for voter in turnout_voters:
                    voter.cast_vote(budget)
                    ballot_box.add_ballot(voter.distribute_coins())

                scores = ballot_box.tally_coin_votes()
                selected_projects_greedy = greedy_selection(valid_combinations, scores, projects)
                cumulative_satisfaction = calculate_cumulative_satisfaction(turnout_voters, selected_projects_greedy, projects)
                total_cumulative_satisfaction += cumulative_satisfaction

                # Save the final scores and selected projects for the last replication
                final_project_scores = scores
                final_selected_projects = selected_projects_greedy

        average_cumulative_satisfaction = total_cumulative_satisfaction / num_replications
        cumulative_greedy_satisfaction.append(average_cumulative_satisfaction)

        # Print the turnout satisfaction and final project details
        print(f"Turnout {turnout}% - Cumulative Greedy Satisfaction: {average_cumulative_satisfaction}")
        print("Final Selected Projects and Details:")
        for project in final_selected_projects:
            project_index = projects.index(project)  # Find the index of the project
            print(f"(Project Name: {project.name}, Cost: {project.cost}, Score: {final_project_scores[project_index]})")

    return turnout_percentages, cumulative_greedy_satisfaction

##################### Plotting Results #######################
def plot_satisfaction(turnout_percentages, cumulative_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(turnout_percentages, cumulative_scores, marker='o', linestyle='-', color='b', label='Greedy')
    plt.title("Cumulative Voter Satisfaction vs. Turnout Percentage (Greedy Algorithm)")
    plt.xlabel("Turnout Percentage (%)")
    plt.ylabel("Cumulative Satisfaction Score")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the simulation and plot results
turnout_percentages, cumulative_greedy_satisfaction = run_simulation()
plot_satisfaction(turnout_percentages, cumulative_greedy_satisfaction)
