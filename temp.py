import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

test_results = pd.read_csv('param_test_results.csv')
savepath = '/home/joe/Code/BioSNICARParameterization/'

plt.figure()
plt.scatter(test_results.snicar_BBA, test_results.param_BBA, marker='^',facecolor='None',color='g')
plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1), color='k')
plt.xlim(0.5,0.75),plt.ylim(0.5,0.75)
plt.ylabel('parameterisation BBA'),plt.xlabel('snicar BBA')
plt.savefig(str(savepath+"BBA_test_results.png"))

plt.figure()
plt.scatter(test_results.snicar_ABS, test_results.param_ABS,marker='o',facecolor='None',color='b')
plt.plot(np.arange(0,200,1),np.arange(0,200,1), color='k')
plt.ylabel('parameterisation abs (W/m2)'),plt.xlabel('snicar abs (W/m2)')
plt.savefig(str(savepath+"ABS_test_results.png"))