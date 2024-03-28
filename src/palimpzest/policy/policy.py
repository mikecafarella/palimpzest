from palimpzest.operators import PhysicalPlan
from palimpzest.sets import Set

from typing import List


class Policy:
    """
    Base class for policies that can choose a best plan from a set of
    candidate plans based on some selection criteria.
    """
    def __init__(self):
        pass

    def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
        pass

class MaxQuality(Policy):
    """
    This policy selects the plan with the maximum quality along the
    pareto-optimal curve of candidate plans.
    """
    def __str__(self):
        return "Maximum Quality"

    def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
        return sorted(candidatePlans, key=lambda cp: cp[2])[-1]


class MinTime(Policy):
    """
    This policy selects the plan with the minimal execution time along the
    pareto-optimal curve of candidate plans.
    """
    def __str__(self):
        return "Minimum Time"

    def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
        return sorted(candidatePlans, key=lambda cp: cp[0])[0]


class MinCost(Policy):
    """
    This policy selects the plan with the minimal cost along the pareto-optimal
    curve of candidate plans.
    """
    def __str__(self):
        return "Minimum Cost"

    def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
        return sorted(candidatePlans, key=lambda cp: cp[1])[0]


class MaxHarmonicMean(Policy):
    """
    This policy selects the plan with the maximum harmonic mean of cost, time, and quality
    along the pareto-optimal curve of candidate plans.
    """
    def __init__(self, max_time: float=600.0, max_cost: float=1.0):
        self.max_cost = max_cost
        self.max_time = max_time

    def __str__(self):
        return "Maximum Harmonic Mean"

    def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
        epsilon = 1e-9
        bestPlan, bestHarmonicMean = None, 0.0
        for plan in candidatePlans:
            # scale time and cost into [0, 1]
            scaled_time = (self.max_time - plan[0]) / self.max_time
            scaled_cost = (self.max_cost - plan[1]) / self.max_cost
            scaled_time = min(max(scaled_time, 0.0), 1.0)
            scaled_cost = min(max(scaled_cost, 0.0), 1.0)

            harmonicMean = 3.0 / ((1.0 / (scaled_time + epsilon)) + (1.0 / (scaled_cost + epsilon)) + (1.0 / plan[2]))

            if harmonicMean > bestHarmonicMean:
                bestHarmonicMean = harmonicMean
                bestPlan = plan

        return bestPlan


class UserChoice(Policy):
    """
    This policy asks the user to decide which of the pareto-optimal
    candidate plans to execute.
    """
    def __str__(self):
        return "User Choice"

    def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
        user_choice = input(f"Please select a plan in [0-{len(candidatePlans) - 1}]: ")
        user_choice = int(user_choice)
        if user_choice not in range(len(candidatePlans)):
            print(f"Error: user choice {user_choice} was not a number in the specified range. Please try again.")
            return self.choose(candidatePlans)

        return candidatePlans[user_choice]
