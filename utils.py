import numpy as np 
import matplotlib.pyplot as plt 


# two plots: 1 for scores and 2 for epsilon 

def plot_learning_curve(x,scores,epsilon,filename): 
    
    fig = plt.figure()
    ax = fig.add_subplot(111,label = "1")
    ax2 = fig.add_subplot(111,label = "2", frame_one=False)
    
    
    # epislon plot 
    
    ax.plot(x,epsilon,color = "C0")
    ax.set_xlabel("Training steps",color = "C0")       
    ax.set_ylabel("Epsilon",color = "C0") 
    ax.tick_params(axis='x',colors = "C0")
    ax.tick_params(axis='y',colors = "C0")

    N = len(scores)
    running_avg = np.empty()
    
    # plotting from 1st to last 100 points of scores - byt getting the avg of these 100's  of data points 
    # Ex: y = 50 -> 0:51 
    # Ex: y = 400 -> 300:401 

    for y in range(N): 
        running_avg = np.mean(scores[max(0,y-100):y+1])
        
    
    ax2.scatter(x,running_avg,color = "C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylabel('Scote',color = "C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y',colors = "C1")
    
    plt.savefig(filename)

        