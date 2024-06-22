import sys
import probability_functions
import data
import numpy as np

def main(n:int):
    #pert_test.generate_pert_distribution_graph(100000,1,4,6)
    samples_arr = []
    titles = []

    cost_samples = probability_functions.stage_simulation(n,data.costs)
    samples_arr.append(cost_samples)
    titles.append('Costs')

    risks_samples = probability_functions.stage_simulation(n, data.risks)
    samples_arr.append(risks_samples)
    titles.append('Risks impact in cost')

    if (isinstance(cost_samples,np.ndarray)  and isinstance(risks_samples,np.ndarray)):
        total_samples = cost_samples + risks_samples
        samples_arr.append(total_samples)
        titles.append('Total cost')

    probability_functions.generate_reports(titles=titles,samples_arr=samples_arr,n=n)
    probability_functions.generate_plots(samples_arr=samples_arr,titles=titles)
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please enter the number of simulations to be run. \n Error: argument missing")
    else:
        main(int(sys.argv[1]))

