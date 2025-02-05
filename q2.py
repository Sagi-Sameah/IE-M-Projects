import heapq
import random
import math


def find_word_path(starting_word, goal_word, search_method, detail_output):
    # Load dictionary
    try:
        with open("dictionary.txt", "r") as f:
            dictionary = set(line.strip().lower() for line in f)
    except FileNotFoundError:
        print("Error: dictionary.txt not found!")
        return

    def is_valid_word(word):
        return word in dictionary

    def heuristic(word, goal):
        cost = 0
        for i in range(max(len(word), len(goal))):
            if i >= len(word):
                cost += 0.25 if goal[i] in "aeiou" else 1
            elif i >= len(goal):
                cost += 0.25 if word[i] in "aeiou" else 1
            elif word[i] != goal[i]:
                cost += 0.25 if word[i] in "aeiou" and goal[i] in "aeiou" else 1
        return cost

    def calculate_cost(current, neighbor):
        return heuristic(current, neighbor)

    def get_neighbors(word):
        neighbors = set()
        for i in range(len(word)):
            for c in "abcdefghijklmnopqrstuvwxyz":
                candidate = word[:i] + c + word[i + 1:]
                if is_valid_word(candidate):
                    neighbors.add(candidate)
        for i in range(len(word) + 1):
            for c in "abcdefghijklmnopqrstuvwxyz":
                candidate = word[:i] + c + word[i:]
                if is_valid_word(candidate):
                    neighbors.add(candidate)
        for i in range(len(word)):
            candidate = word[:i] + word[i + 1:]
            if is_valid_word(candidate):
                neighbors.add(candidate)
        return neighbors

    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    if search_method == 1:  # A* Search
        open_set = []
        heapq.heappush(open_set, (0, starting_word))
        came_from = {}
        g_score = {starting_word: 0}
        f_score = {starting_word: heuristic(starting_word, goal_word)}

        while open_set:
            _, current_word = heapq.heappop(open_set)
            if current_word == goal_word:
                path = reconstruct_path(came_from, current_word)
                total_heuristic = heuristic(starting_word, goal_word)
                if detail_output:
                    print(f"Heuristic: {total_heuristic:.2f}")
                    print("\n".join(path))
                return path

            for neighbor in get_neighbors(current_word):
                tentative_g_score = g_score[current_word] + calculate_cost(current_word, neighbor)
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current_word
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_word)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print("No path found")
        return

    elif search_method == 2:  # Improved Hill Climbing
        max_restarts = 5
        for restart in range(max_restarts):
            current_word = starting_word
            path = [current_word]
            while True:
                neighbors = get_neighbors(current_word)
                if not neighbors:
                    break
                next_word = min(neighbors, key=lambda n: heuristic(n, goal_word), default=None)
                if next_word is None or heuristic(next_word, goal_word) >= heuristic(current_word, goal_word):
                    break
                current_word = next_word
                path.append(current_word)
            if current_word == goal_word:
                total_heuristic = heuristic(starting_word, goal_word)
                print(f"Heuristic: {total_heuristic:.2f}")
                print("\n".join(path))
                return
        print("No path found")

    elif search_method == 3:  # Simulated Annealing
        current_word = starting_word
        t = 100
        cooling_rate = 0.9  # Gradual cooling for better convergence
        path = [current_word]
        while t > 1:
            neighbors = list(get_neighbors(current_word))
            if not neighbors:
                break
            next_word = random.choice(neighbors)
            delta_e = heuristic(current_word, goal_word) - heuristic(next_word, goal_word)
            probability = min(1, math.exp(delta_e / t)) if delta_e < 0 else 1.0
            if random.random() < probability:
                current_word = next_word
                if current_word == goal_word:
                    path.append(current_word)
                    print("Goal reached!")
                    if detail_output:
                        print("\n".join(path))
                    return path
            t *= cooling_rate

            print("No path found")
            return

    elif search_method == 4:  # Improved Local Beam Search
        k = 3
        beams = [starting_word]
        max_iterations = 20
        iteration = 0
        while beams and iteration < max_iterations:
            all_neighbors = []
            for word in beams:
                all_neighbors.extend(get_neighbors(word))
            if goal_word in all_neighbors:
                print(goal_word)
                return
            unique_neighbors = list(set(all_neighbors))
            unique_neighbors.sort(key=lambda n: heuristic(n, goal_word))
            beams = unique_neighbors[:k]
            print(f"Bag of actions: {beams}")
            iteration += 1
        print("No path found")

    elif search_method == 5:  # Genetic Algorithm
        population = [starting_word] * 10
        for generation in range(100):
            new_population = []
            for word in population:
                if random.random() < 0.2:  # Mutation
                    word = random.choice(list(get_neighbors(word)))
                new_population.append(word)
            population = sorted(new_population, key=lambda n: heuristic(n, goal_word))[:10]
            print(f"Generation {generation}:", population)
            if goal_word in population:
                print("Path found!")
                if detail_output:
                    print(f"Final Population: {population}")
                return population

        print("No path found")
        return

# Example Usage
if __name__ == "__main__":
    try:
        result = find_word_path("dog", "cat", 1, True)
    except Exception as e:
        print(f"An error occurred: {e}")
