import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from KalmanFilter import *
import os

# Data
FundNames = ["Placeholder","LEBUAGCH","LEBUCGCH","LEBUPFCH","LEBUWPCH"]
FundNo = 2 # The above has 4 funds, we use placeholder to shift indexes nicely according to Fun1,...,Fund4
file = "FundAndIndexData.csv" # This file has formate Dates, Proxy1, Proxy2,..,Fund1,Fund2,Fund3,Fund4
NumProxies = 2 # Leave this at 2 unless you add or remove proxy indicies


def read_csv_file(file_name):
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go one directory up
    parent_dir = os.path.dirname(script_dir)
    print(parent_dir)
    
    # Define the path to the "Data" folder in the parent directory
    data_folder = os.path.join(parent_dir, "Data")
    
    # Construct the full file path
    file_path = os.path.join(data_folder, file_name)
    
    # Read the CSV file
    if os.path.exists(file_path):
        return pd.read_csv(file_path, skiprows=1)
    else:
        raise FileNotFoundError(f"File '{file_name}' not found in {data_folder}")
    
def read_txt_file(file_name, **kwargs):
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go one directory up
    parent_dir = os.path.dirname(script_dir)
    
    # Define the path to the "Data" folder in the parent directory
    data_folder = os.path.join(parent_dir, "Saved States")
    
    # Construct the full file path
    file_path = os.path.join(data_folder, file_name)
    
    # Read the TXT file using numpy
    if os.path.exists(file_path):
        return np.loadtxt(file_path, **kwargs)
    else:
        raise FileNotFoundError(f"File '{file_name}' not found in {data_folder}")

def readData(data, NumProxies, FundNo):
    X = data.iloc[:,1:(1+NumProxies)]
    Xnames = X.columns
    Dates = data.iloc[:,0]
    X = np.transpose(X.to_numpy())
    Y = data.iloc[:,NumProxies+FundNo]
    Y = Y*100_000_000
    FundName = Y.name
    Y = np.array([[y for y in np.transpose(Y.to_numpy())]])
    return X,Y,Xnames,FundName,Dates 

data = read_csv_file(file)
# print(data)
X,Y,ProxyNames,Fund,Dates = readData(data,NumProxies,FundNo)

print(Dates)

# Random Set-up
d = X.shape[0]
n = X.shape[1]
xs = [X[:,k] for k in range(0,n)]
ys = [Y[:,k] for k in range(0,n)]

state_0 = read_txt_file(f"state0.txt")
state_0 = np.reshape(state_0,(d,1))
# print(state_0)
P_0 = read_txt_file(f"P0.txt")
# print(P_0)
Q = read_txt_file("Q.txt")
# print(Q)
R = read_txt_file("R.txt")
R = np.reshape(R,(1,1))
# print(R)

states = [state_0]
errs = [0]
state = state_0
Ps = [P_0]

for k in range(1,len(xs)):
	x = xs[k]
	y = ys[k]
	(state,P,err) = Kalman(x,y,state,Ps[-1],Q,R,d)
	state = state.reshape((d,1))
	Ps += [P]
	states += [state]
	errs+=[err.flatten()[0]]
	Date = Dates.iloc[k]#.strftime('%Y-%m-%d')

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
final_P = Ps[-1]
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