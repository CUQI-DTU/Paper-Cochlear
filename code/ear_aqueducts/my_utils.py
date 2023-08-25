import numpy as np
import matplotlib.pyplot as plt

def plot_time_series(times, locations, data):

    # Plot data
    color = ['r', 'g', 'b', 'k', 'm', 'c']
    legends = ['loc = '+"{:.2f}".format(obs) for obs in locations]
    lines = []
    for i in range(len(locations)):
        lines.append(plt.plot(times/60, data[i,:],  color=color[i])[0])
    
    plt.legend(lines, legends)
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration')
