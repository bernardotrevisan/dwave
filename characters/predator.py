import numpy as np
from random import randint, seed

class Predator:
    """
    The Predator class represents the predator in the Predator-Prey task.

    ...

    Attributes
    ----------
    loc : [float]
        Location of the predator [x, y]
    feasted : bool
        Says whether the predator has caught the agent or not
    loc_trace : [[float]]
        Keeps track of all the locations the predator was in
    w : int
        Width of the coordinate plane
    h : int
        Height of the coordinate plane

    Methods
    -------
    pursue(agent_loc, speed)
        Pursues the agent given its location and a speed of movement.
    bounce_back()
        If the predator's location is outside the coordinate plane, bounce back into it.
    """

    def __init__(self, w, h):
        """
        Parameters
        ----------
        w : int
            Width of the coordinate plane
        h : int
            Height of the coordinate plane
        """
        seed(2)
        self.loc = [randint(int(2*w/3), w), randint(0, h)]
        self.feasted = False
        self.loc_trace = [list(self.loc)]
        self.w = w
        self.h = h
        return

    def pursue(self, agent_loc, speed):
        """Pursues the agent given its location and a speed of movement

        Parameters
        ----------
        agent_loc : [float]
            The agent's location [x, y]
        speed : float
            The speed of movement

        Returns
        -------
        void

        Raises
        ------
        ValueError
            If given arguments are invalid.
        """

        if agent_loc is None:
            raise ValueError("agent_loc must be valid")

        if speed <= 0:
            raise ValueError("speed must be positive number")

        # If the distance between prey and predator is less than 10 it counts as a contact
        buffer = 10
        # Vector for the predator's location
        pred_v = np.array(self.loc)
        # Vector for the agent's  location
        agent_v = np.array(agent_loc)
        # Vector for the direction of movement
        move_v = agent_v - pred_v
        # Distance between predator and agent
        dist2agent = np.linalg.norm(move_v)

        # Move predator alongside this vector at a given speed
        d = speed / dist2agent
        if d > 1:
            d = 1
        new_loc = np.floor((pred_v + d * move_v))

        # Update prdator's location
        self.loc = new_loc

        # Update distance to agent
        dist2agent = np.linalg.norm(agent_loc - np.array(self.loc))
        # If the agent has been caught, set feasted to True
        if dist2agent < buffer:
            self.feasted = True

        # Update location trace
        self.loc_trace.append(list(self.loc))
        return
    
    def __repr__(self):
        """Displays information about the predator
        """
        display = ['\n===============================']
        display.append('P R E D A T O R')
        display.append('Feasted: ' + str(self.feasted))
        display.append('Steps taken: ' + str(len(self.loc_trace)))
        display.append('Location trace:')
        loc_trace_str = ""
        for loc in self.loc_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            loc_trace_str += ", " + str(loc)
        display.append(loc_trace_str)
        display.append('===============================\n')
        return "\n".join(display)
