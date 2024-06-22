import numpy as np
import matplotlib.pyplot as plt
import data

costs = data.costs
risks = data.risks

def montecarlo_simulation(n:int, m:int):
    """
    Perform a montecarlo simulation for secuential events based on random samples taken from a PERT distribution

    Parameters:
    - n (int): number of simulations 
    """
