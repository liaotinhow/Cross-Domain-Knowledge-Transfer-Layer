import matplotlib.pyplot as plt
import ipdb
import numpy as np
cmap = plt.get_cmap('gnuplot')
n_act=np.array([5,4,58,7,8,13,20,6])#np.array([3,1,2,4,5,6,7,8,9,10,11,12,13,20,14,15,18,19,21,22,2    3,24,25,26,27,28,29,30,31,32,33,34,35,37,38,40,43,44,45,46,47,50,51,52,53,54,55,56,58])#np.array([7,8,5,13,14,2    7,0,22,9])
colors = [cmap(i) for i in np.linspace(0,1,9)]
ipdb.set_trace()
