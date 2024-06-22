import random
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

def rpert(n, x_min, x_mode, x_max, lambda_=4):
    """
    Generate random samples from a PERT distribution.
    
    Parameters:
    - n (int): Number of samples to generate.
    - x_min (float): The minimum value of the distribution.
    - x_max (float): The maximum value of the distribution.
    - x_mode (float): The most likely (mode) value of the distribution.
    - lambda_ (float): The shape scaling parameter (default 4).
    
    Returns:
    - numpy.ndarray: Array of random samples from the specified PERT distribution.
    """
    
    # Validate the parameters passed
    if x_min > x_max or x_mode > x_max or x_mode < x_min:
        print(f'min: {x_min}, mode: {x_mode}, max: {x_max}')
        raise ValueError("Invalid parameters: ensure x_min <= x_mode <= x_max")
    
    x_range = x_max - x_min
    if x_range == 0:
        return np.full(n, x_min)
    
    # Calculate the mean mu
    mu = (x_min + x_max + lambda_ * x_mode) / (lambda_ + 2)
    
    # Calculate v and w parameters used to shape the distribution
    if mu == x_mode:
        v = lambda_ / 2 + 1
    else:
        v = ((mu - x_min) * (2 * x_mode - x_min - x_max)) / ((x_mode - mu) * (x_max - x_min))
        w = (v * (x_max - mu)) / (mu - x_min)
    
    # Use the beta distribution to generate n samples and scale them to the PERT range
    samples = beta.rvs(v, w, size=n)
    return samples * x_range + x_min

def event(probability:float):
    """
    Sample a binary event based on a given probabilty

    Parameters: 
    - probability (float): The probability of the event ocurring, must be between 0 and 1 (both inclusive)

    Returns:
    - int: 1 if the event occurs, 0 otherwise
    """
    if not(0 <= probability <= 1):
        return ValueError("Probability must be between 0 and 1")
    
    return 1 if random.random() < probability else 0


def stage_simulation(n:int, data:dict): 
    """
    Perform a montecarlo simulation for sequential events based on random samples taken from a PERT distribution

    Parameters:
    - n (int): number of simulations 
    - data (dict): the data from which generate the samples

    Returns: 
    - list[float]: the sum of the sequential tasks whose duration  is sampled from the PERT distribution
    """
    import os
    import subprocess

    results = []
    for task in data.keys():
        occurrences= [event(data[task]["probability"]) for x in range(n)]
        values = rpert(n, data[task]["values"][0],data[task]["values"][1],data[task]["values"][2])
        results.append([x * y for x, y in zip(occurrences, values)]) 
    
    totals = np.sum(results, axis=0)
    
    # Return the samples generated
    return totals

def generate_plots(samples_arr:np.ndarray, titles:list[str]):
    """
    Generate plots for each of the group of samples provided

    Parameters: 
    - samples_arr (np.ndarray): array containing groups of samples
    - title (str): title of the plot

    Returns: None
    """ 
    def add_stats(ax, data, color_mean, color_std):
        mean = np.mean(data)
        std = np.std(data)
        # Add a vertical line for the mean
        ax.axvline(mean, color=color_mean, linestyle='dashed', linewidth=2)
        # Add vertical lines for one standard deviation on either side of the mean
        ax.axvline(mean - std, color=color_std, linestyle='dotted', linewidth=2)
        ax.axvline(mean + std, color=color_std, linestyle='dotted', linewidth=2)

        # Annotate the mean and standard deviation
        ax.text(mean, plt.ylim()[1]*0.9, f'Mean: {mean:.2f}', horizontalalignment='center', color='black')
        ax.text(mean - std, plt.ylim()[1]*0.8, f'-1σ: {mean-std:.2f}', horizontalalignment='center', color='black')
        ax.text(mean + std, plt.ylim()[1]*0.8, f'+1σ: {mean+std:.2f}', horizontalalignment='center', color='black')

    for i, samples in enumerate(samples_arr):
        # visualization of the stage samples distribution
        plt.figure(i)
        ax = plt.subplot(111)
        ax.hist(samples, bins=1000, alpha=0.75)
        add_stats(ax, samples, 'green', 'red')
        plt.title(f"{titles[i]} probability distribution")
        plt.xlabel("Sum")
        plt.ylabel("Frequency")
    plt.show()

def generate_report(title:str, mean:float, std:float, n:int):
    """
    Generate a report of a given distribution\
    
    Parameters:
    - title (str): the title of the report
    - mean (float): the mean of the distribution 
    - std (float): the standard deviation of the distribution
    - n (int): number of simulations used to generate the distribution

    Returns: None
    """
    import os
    import subprocess

    output_file_name = fr'reports\{title}_output.txt'
    with open(output_file_name, 'w') as file:
        # Header
        file.write(f"{f'{title} output Report':^50}\n")
        file.write(f"{'='*50}\n")
        
        # Content with formatting for better readability
        file.write(f"{'Number of simulations:':<30}{n:>20}\n")
        file.write(f"{'Mean of distribution:':<30}{mean:>20.2f}\n")
        file.write(f"{'Standard deviation:':<30}{std:>20.2f}\n")
        
        one_sigma= mean - std if mean - std >= 0 else 0
        two_sigma= mean - 2 * std if mean - 2 * std >= 0 else 0
        three_sigma= mean - 3 * std if mean - 3 * std >= 0 else 0

        file.write(f"\n{f'Confidence Intervals for {title}:':<30}\n")
        file.write(f"{'-'*50}\n")
        file.write(f"{'68% (1 sigma)':<30}{one_sigma:>10.2f} - {mean + std:<10.2f}\n")
        file.write(f"{'95% (2 sigma)':<30}{two_sigma:>10.2f} - {mean + 2 * std:<10.2f}\n")
        file.write(f"{'99.7% (3 sigma)':<30}{three_sigma:>10.2f} - {mean + 3 * std:<10.2f}\n")
        
        # Footer
        file.write(f"{'='*50}\n")
        file.write(f"{'End of Report':^50}\n")

    # Confirm to the user that the data has been written
    print("Output has been written to output.txt")
    print(f"Opening {title}_report.txt") 

    #abs_path = os.path.join(os.getcwd(), output_file_name) 
    os.startfile(output_file_name)
    #subprocess.Popen(['start', abs_path])    

def generate_reports(titles:list, samples_arr:list, n:int):
    """
    Generate statistical reports of each of the samples provided

    Parameters: 
    - titles (list[str]): a list of titles for the corresponding reports
    - samples_arr (list[np.ndarray]): a list of np.ndarray containing the samples from the montecarlo simulations
    - n (int): the number of simulations used to generate the samples

    Returns: None
    """
    for i, samples in enumerate(samples_arr):
        mean = np.mean(samples)
        std = np.std(samples)
        generate_report(titles[i], mean, std, n)

