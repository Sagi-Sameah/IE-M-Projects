import heapq

def find_word_path(starting_word, goal_word, search_method, detail_output):
    # Load dictionary
    try:
        with open('dictionary.txt', 'r') as f:
            dictionary = set(line.strip().lower() for line in f)
    except FileNotFoundError:
        print("Error: dictionary.txt not found!")
        return

    def is_valid_word(word):
        return word in dictionary

    def heuristic(word, goal):
        """
        General heuristic function incorporating the prefix length heuristic
        but ultimately using the cost-based heuristic for better estimation.
        """
        common_prefix_length = 0
        for a, b in zip(word, goal):
            if a == b:
                common_prefix_length += 1
            else:
                break

        prefix_heuristic = len(goal) - common_prefix_length
        cost_heuristic = calculate_cost(word, goal)
        if prefix_heuristic < cost_heuristic:
            return prefix_heuristic
        return cost_heuristic

    def calculate_cost(current, neighbor):
        cost = 0
        for i in range(max(len(current), len(neighbor))):
            if i >= len(current):  # Addition
                cost += 0.25 if neighbor[i] in 'aeiou' else 1
            elif i >= len(neighbor):  # Removal
                cost += 0.25 if current[i] in 'aeiou' else 1
            elif current[i] != neighbor[i]:
                if current[i] in 'aeiou' and neighbor[i] in 'aeiou':
                    cost += 0.25
                else:
                    cost += 1

        return cost

    def get_neighbors(word):
        neighbors = set()
        # Substitution
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i + 1:]
                if is_valid_word(new_word):
                    neighbors.add(new_word)
        for i in range(len(word) + 1):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i:]
                if is_valid_word(new_word):
                    neighbors.add(new_word)
        for i in range(len(word)):
            new_word = word[:i] + word[i + 1:]
            if is_valid_word(new_word):
                neighbors.add(new_word)
        return neighbors

    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    open_set = []
    heapq.heappush(open_set, (0, starting_word))
    came_from = {}
    g_score = {starting_word: 0}
    f_score = {starting_word: heuristic(starting_word, goal_word)}
    explored_nodes = 0
    max_nodes = 10000

    while open_set:
        if explored_nodes > max_nodes:
            print("No path found")

        current_cost, current_word = heapq.heappop(open_set)
        explored_nodes += 1
        if current_word == goal_word:
            path = reconstruct_path(came_from, current_word)
            total_heuristic = heuristic(starting_word, goal_word)  # Keep total heuristic
            if detail_output:
                print(f"Heuristic: {total_heuristic:.2f}")
                print("\n".join(path))
            else:
                print("\n".join(path))
            return path

        for neighbor in get_neighbors(current_word):
            tentative_g_score = g_score[current_word] + calculate_cost(current_word, neighbor)
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current_word
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_word)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return "No path found"

# Example Usage
if __name__ == "__main__":
    try:
        result = find_word_path("dog", "cat", 1, True)
    except Exception as e:
        print(f"An error occurred: {e}")
