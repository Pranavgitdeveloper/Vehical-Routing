# ---- Required Libraries ----
import numpy as np                    # For numerical operations and arrays
import matplotlib.pyplot as plt      # For plotting the final vehicle routes
import random                        # For randomization in the genetic algorithm
import os                            # For creating folders and saving output

# ---- Calculate Euclidean Distance between two points ----
def distance(a, b):
    return np.linalg.norm(a - b)

# ---- Total distance for a given vehicle route ----
def route_distance(route, depot, customers):
    dist = 0
    current = depot  # Start from depot
    for idx in route:
        dist += distance(current, customers[idx])  # Add distance to next customer
        current = customers[idx]
    dist += distance(current, depot)  # Return to depot at end
    return dist

# ---- Split chromosome into multiple vehicle routes ----
def split_routes(chromosome, num_vehicles):
    avg = len(chromosome) // num_vehicles  # Evenly divide customers among vehicles
    return [chromosome[i*avg:(i+1)*avg] for i in range(num_vehicles)]

# ---- Calculate total fitness (i.e., total distance) ----
def fitness(chromosome, depot, customers, num_vehicles):
    routes = split_routes(chromosome, num_vehicles)
    return sum(route_distance(route, depot, customers) for route in routes)

# ---- Generate initial random population ----
def generate_population(num_customers, size):
    base = list(range(num_customers))  # Customer indices
    return [random.sample(base, num_customers) for _ in range(size)]  # Shuffle for diversity

# ---- Perform crossover (recombination) between two parents ----
def crossover(parent1, parent2):
    size = len(parent1)
    p1, p2 = sorted(random.sample(range(size), 2))  # Two crossover points
    middle = parent1[p1:p2]                         # Take slice from parent1
    rest = [item for item in parent2 if item not in middle]  # Maintain order from parent2
    return rest[:p1] + middle + rest[p1:]  # Combine to form new child

# ---- Mutate a chromosome with given mutation rate ----
def mutate(chromosome, rate):
    if random.random() < rate:
        a, b = random.sample(range(len(chromosome)), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]  # Swap two genes
    return chromosome

# ---- Main function to solve Vehicle Routing Problem ----
def solve_vrp(num_customers, customers_list, num_vehicles, depot_list):
    # Convert input lists to NumPy arrays
    customers = np.array(customers_list)
    depot = np.array(depot_list)

    # Genetic Algorithm parameters
    POP_SIZE = 50
    GENERATIONS = 100
    MUTATION_RATE = 0.1

    # Initialize population
    population = generate_population(num_customers, POP_SIZE)

    # Run Genetic Algorithm for multiple generations
    for _ in range(GENERATIONS):
        population.sort(key=lambda c: fitness(c, depot, customers, num_vehicles))  # Sort by fitness
        new_pop = population[:5]  # Elitism: Keep top 5 solutions

        # Generate new population via crossover and mutation
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.choices(population[:25], k=2)  # Select parents from top 25
            child = crossover(p1, p2)
            child = mutate(child, MUTATION_RATE)
            new_pop.append(child)

        population = new_pop  # Update population

    # Select best solution after all generations
    best = min(population, key=lambda c: fitness(c, depot, customers, num_vehicles))
    best_routes = split_routes(best, num_vehicles)
    total_distance = fitness(best, depot, customers, num_vehicles)

    # ---- Plotting the Result ----
    colors = ['r', 'g', 'b', 'c', 'm']  # Colors for each vehicle
    plt.figure(figsize=(8, 6))

    # Plot depot
    plt.scatter(depot[0], depot[1], c='k', marker='s', label='Depot')

    # Plot each customer
    for i, cust in enumerate(customers):
        plt.scatter(cust[0], cust[1], label=f'C{i}')

    # Plot each vehicle route
    for i, route in enumerate(best_routes):
        x = [depot[0]] + [customers[idx][0] for idx in route] + [depot[0]]
        y = [depot[1]] + [customers[idx][1] for idx in route] + [depot[1]]
        plt.plot(x, y, color=colors[i % len(colors)], label=f'Route {i+1}')

    plt.title('Optimized Vehicle Routes')
    plt.legend()
    plt.grid(True)

    # Save the figure to static folder for display on website
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/plot.png")
    plt.close()

    # ---- Prepare text result to return ----
    result_text = "Best Routes Found:\n"
    for i, route in enumerate(best_routes):
        result_text += f"Vehicle {i+1}: {route}\n"
    result_text += f"Total Distance: {total_distance:.2f}"

    return result_text
