import numpy as np
from cmath import *
from scipy.stats import norm

# Kalman Code
def Kalman(x,y,state,P,Q,R,d):
	#x is one obs from X and y is one obs from Y
	#F = Identity
	P+=Q
	H = np.array([x])
	z = np.array([[y]])
	err = z-np.matmul(H,state)
	S = np.matmul(H,np.matmul(P,np.transpose(H)))+R
	K = np.matmul(P,np.transpose(H))
	K = np.matmul(K,np.linalg.inv(S))
	state = state+np.matmul(K,err)
	P = np.matmul((np.eye(d)-np.matmul(K,H)),P)
	err = z-np.matmul(H,state)
	return (state,P,err)

def Kalman_run(xs,ys,state_0,P_0,Q,R,d):
	states = [state_0]
	errs = [0]
	state = state_0
	P = P_0
	L=1
	for k in range(0,len(xs)):
		x = xs[k]
		y = ys[k]
		(state,P,err) = Kalman(x,y,state,P,Q,R,d)
		L*=norm.pdf(err)
		states.append(state)
		errs+=[err.flatten()[0]]
	return (L,states,errs)

def set_params(params,d):
	state_0 = np.reshape(params[0:d],(d,1))
	k = d
	P_0 = np.zeros((d,d))
	for ii in range(0,d):
		for jj in range(0,d):
			if ii>=jj:
				P_0[ii,jj] = params[k]
				k+=1
	P_0 = np.matmul(P_0,np.transpose(P_0))
	Q = np.zeros((d,d))
	for ii in range(0,d):
		for jj in range(0,d):
			if ii>=jj:
				Q[ii,jj] = params[k]
				k+=1
	Q = np.matmul(Q,np.transpose(Q))
	R = np.array([[params[k]**2]])

	return (state_0,P_0,Q,R)