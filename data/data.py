import numpy as np
import math

class Data:
    def __init__(self, size, max_speed, w, h):
        self.size = size
        self.max_speed = max_speed
        self.w = w
        self.h = h
        self.data = self.generate_data()

    # Generate data for the network
    def generate_data(self):
        # Each datapoint follows the format [prey_loc, agent_loc, predator_loc, best_loc]
        data = []
        for _ in range(self.size):
            datapoint = []
            # Generate random positions for the agent, prey, and predator
            # Prey's position
            datapoint.append(self.generate_random_loc(0, self.w/3))
            # Agent's position
            datapoint.append(self.generate_random_loc(self.w/3, 2*self.w/3))
            # Predator's position
            datapoint.append(self.generate_random_loc(2*self.w/3, self.w))

            # Get the best location the agent could move to
            datapoint.append(self.best_loc(datapoint))

            data.append(datapoint)
        
        return data

    # Generate a random position given the lower and the upper bounds on the x-axis
    def generate_random_loc(self, lower, upper):
        return [np.random.randint(lower, upper+1), np.random.randint(0, self.h)]

    def best_loc(self, positions):
        # Let the position of the agent be the center of the circle of radius equal to max_speed
        center = positions[1]
        radius = self.max_speed

        # All possible angles along the circumference
        angles = [x for x in range(361)]
        best_reward = -math.inf
        best_loc = None

        for angle in angles:
            # Get the x and y coordinates if the agent were to move at this angle
            x = radius * np.cos(np.radians(angle)) + center[0]
            if x < 0:
                x = 0
            elif x > self.w:
                x = self.w

            y = radius * np.sin(np.radians(angle)) + center[1]
            if y < 0:
                y = 0
            elif y > self.h:
                y = self.h

            loc = [x, y]

            # Reward = distance to predator - distance to prey
            reward = math.dist(loc, positions[2]) - math.dist(loc, positions[0])
            # If this locations has the best reward seen yet, update
            if reward > best_reward:
                best_reward = reward
                best_loc = loc
    
        return best_loc
