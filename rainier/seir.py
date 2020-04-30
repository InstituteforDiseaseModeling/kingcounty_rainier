""" seir.py

SEIR model class and associated tools. This model is the most basic, with no spatial
correlation or connectivity. For more details, see the doc-string of the model class. 
The model is loosely based on We et al, Lancet, 2019, so that might also be a good place
to look. """
import sys
import numpy as np

## Statistics tools
from rainier.splines import SmoothingSpline
from rainier.stats import WeightedLeastSquares

## For model fitting
from scipy.optimize import minimize

## For good hygiene
import warnings

class LogNormalSEIR(object):

	""" Discrete time SEIR model with log-normally distributed transmission. 

	S_t = S_{t-1} - E_{t-1}
	E_t = beta*S_{t-1}*(I_{t-1}+z_{t-1})*epsilon_t + (1-(1/D_e))*E_{t-1}
	I_t = (1/D_e)*E{t-1} + (1-(1/D_i))*I_{t-1}
	C_t ~ Binomial(I_t,p) 

	D_e, D_i, z_t, and the initial condition are assumed known. beta, alpha, p, and 
	transmission variance are meant to be inferred from data. """

	def __init__(self,S0,D_e,D_i,z_t):

		## Store the known model parameters
		self.z_t = z_t
		self.S0 = S0
		self.D_e = D_e
		self.D_i = D_i

		## Create a time axis
		self.T = len(z_t)
		self.time = np.arange(self.T)

		## Mark the model as un-fit, which means parameters
		## are missing.
		self._fit = False

	def sample(self,beta_t,num_samples=10000,sig_eps=None,z_t=None):

		""" Sample the model, num_samples is the number of sample trajectories. Output has shape
		(num_samples,4,len(beta_t)). """

		## Set the variance for these samples
		if sig_eps is None:
			sig_eps = self.sig_eps*np.ones((len(beta_t),))

		## Set the importation scenario
		if z_t is None:
			z_t = self.z_t

		## Allocate storage for the output, and set up the
		## initial condition.
		X = np.zeros((num_samples,3,len(beta_t)))
		X[:,0,0] = self.S0

		## Loop over time and collect samples
		for t in range(1,len(beta_t)):

			## Sample eps_t
			eps_t = np.exp(np.random.normal(0,sig_eps[t],size=(num_samples,)))

			## Update all the deterministic components (S and I)
			X[:,0,t] = X[:,0,t-1]-beta_t[t]*X[:,0,t-1]*(X[:,2,t-1]+z_t[t-1])*eps_t
			X[:,2,t] = X[:,1,t-1]/self.D_e + X[:,2,t-1]*(1.-(1./self.D_i))

			## Update the exposed compartment accross samples
			X[:,1,t] = beta_t[t]*X[:,0,t-1]*(X[:,2,t-1]+z_t[t-1])*eps_t+\
					   X[:,1,t-1]*(1.-(1./self.D_e))

			## High sig-eps models require by-hand enforcement
			## of positivity (i.e. truncated gaussians).
			X[X[:,2,t]<0,2,t] = 0
			X[X[:,1,t]<0,1,t] = 0
			X[X[:,0,t]<0,0,t] = 0

		return X

	def mean(self,beta_t,z_t=None,sig_eps=None):

		""" Compute the mean trajectory given a time-varying beta_t series. """

		## Set the importation scenario
		if z_t is None:
			z_t = self.z_t

		## Set the variance over time
		if sig_eps is None:
			sig_eps = self.sig_eps*np.ones((len(beta_t),))

		## Allocate storage for the output, and set up the
		## initial condition.
		X = np.zeros((3,len(beta_t)))
		X[0,0] = self.S0

		## Loop over time and collect samples
		eps_t = np.exp(0.5*sig_eps**2)
		for t in range(1,len(beta_t)):

			## Update all the deterministic components (all of them in this case)
			X[0,t] = X[0,t-1]-beta_t[t]*X[0,t-1]*(X[2,t-1]+z_t[t-1])*eps_t[t]
			X[2,t] = X[1,t-1]/self.D_e + X[2,t-1]*(1.-(1./self.D_i))
			X[1,t] = beta_t[t]*X[0,t-1]*(X[2,t-1]+z_t[t-1])*eps_t[t]+\
					   X[1,t-1]*(1.-(1./self.D_e))

		return X

def SampleMortality(model,samples,ifr_samples,delay_samples):

	""" Sample mortality accross samples of ifr and delays. """

	## Calculate new exposures
	new_exposures = samples[:,1,1:] - (1. - (1./model.D_e))*samples[:,1,:-1]

	## Compute destined deaths with a different IFR for each
	## trajectory.
	destined_deaths = np.random.binomial(np.round(new_exposures).astype(int),
										 p=ifr_samples[:,np.newaxis])

	## Finally, use np.roll to shift by appropriate numbers, and then
	## zero rolled entries for daily deaths (this is the slow part!)
	daily_deaths = advindexing_roll(destined_deaths,delay_samples)
	for i,d in enumerate(delay_samples):
		daily_deaths[i,:d] = 0

	return destined_deaths, daily_deaths

def advindexing_roll(A, r):
	
	""" The following code was originally posted to StackOverflow as part of 
	question https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently. 
	Contributing authors were seberg (https://stackoverflow.com/users/455221/seberg), 
	yatu (https://stackoverflow.com/users/9698684/yatu), 
	divakar (https://stackoverflow.com/users/3293881/divakar), 
	Yann Dubois (https://stackoverflow.com/users/6232494/yann-dubois), and 
	logicOnAbstractions (https://stackoverflow.com/users/3633696/logiconabstractions). 
	Code from StackOverflow is provided subject to the Creative Commons Attribution Share-Alike license. """
	
	rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]
	r[r < 0] += A.shape[1]
	column_indices = column_indices - r[:,np.newaxis]
	return A[rows, column_indices]
