import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from KalmanFilter import *
from utils import read_xlsx_file, read_txt_file, readData, generate_initial_states


# Utils init
file = "Combined_data.xlsx"
Fund = "QQQ"
ProxyCols = ["amazon", "google"]  # Adjust based on Combined_data.xlsx
NumProxies = len(ProxyCols)
data = read_xlsx_file(file)

# Extract columns
Dates = data["Date"]
X = data[ProxyCols].to_numpy().T
Y_raw = data[Fund] * 100_000_000
FundName = Fund
Y = np.array([[y for y in Y_raw.to_numpy().T]])
ProxyNames = np.array(ProxyCols)

# Random Set-up
d = X.shape[0]
n = X.shape[1]
xs = [X[:,k] for k in range(0,n)]
ys = [Y[:,k] for k in range(0,n)]

state_0, P_0, Q, R = generate_initial_states(NumProxies, scale=1.0, overwrite=True)

_, states, errs = Kalman_run(xs, ys, state_0, P_0, Q, R, d) # we don't care about the likelihood value right now, so don't store

plt.subplot(221)
plt.plot(errs)
plt.title("Fund holdings error through time")

FundVals = Y[0]/100_000_000
weights = [states[i].flatten()/100_000_000 for i in range(1,n)]
FundRep = np.sum([X[:,i+1]*weights[i] for i in range(n-1)],axis=1)
FundToPlot = FundVals[-101:-1]
FundRepToPlot = FundRep[-100:] # This is only shifted so that I can see the difference
# print(FundRep)
plt.subplot(212)
plt.plot(FundToPlot)
plt.plot(FundRepToPlot)
plt.title("Fund Replication for %s (offset by one day to see shape)"%(Fund))
plt.legend([Fund+" Fund Value","Replication Portfolio"]) # type: ignore

s = weights#[np.squeeze(states[i])/100_000_000 for i in range(1, n)]
plt.subplot(222)
plt.plot(s)
plt.title("Weights for constituents for %s"%(Fund))
plt.tight_layout()
plt.show()

second_Last_Date = "2024-02-16"
#final_P = Ps[-1] ##REMINDER TO COME BACK TO THIS
final_state = states[-1]
#test_P = np.loadtxt(f"C:/Users/X493548/OneDrive - Old Mutual/Desktop/OMSFIN Ready Hedges/Saved States/{Fund} States/{second_Last_Date}_P_{Fund}.txt")
#test_state = np.loadtxt(f"C:/Users/X493548/OneDrive - Old Mutual/Desktop/OMSFIN Ready Hedges/Saved States/{Fund} States/{second_Last_Date}_state_{Fund}.txt")
#test_state = np.reshape(test_state,(d,1))
#(state,P,err) = Kalman(xs[-1],ys[-1],test_state,test_P,Q,R,d)
# print(P==final_P)
# print(state==final_state)

print(f"Date: {Dates.iloc[-1]}")
ls = [f"{name}:{weight}" for name,weight in zip(ProxyNames,weights[-1])]
print(f"The weights for fund {Fund} are: {ls}")
# %%
