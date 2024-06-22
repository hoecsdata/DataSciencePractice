import probability_functions
import numpy as np
import matplotlib.pyplot as plt

def generate_pert_distribution_graph(n, x_min, x_mode, x_max):
    samples = probability_functions.rpert(n,x_min=x_min, x_mode=x_mode, x_max=x_max)
    plt.figure(figsize=(10,6))
    plt.hist(samples, bins=300, color='blue', alpha=0.7, label='PERT Distribution')
    plt.title('Histogram of PERT distribution samples')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
