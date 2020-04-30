""" seir.py

SEIR model class and associated tools. This model is the most basic, with no spatial
correlation or connectivity. For more details, see the doc-string of the model class. 
The model is loosely based on We et al, Lancet, 2019, so that might also be a good place
to look. 

Notes:
1. Model is unstable if there's an alpha term? Does that have to do with daily time
steps?
2. What contributions should the spline smoothing terms give to the -logposterior? 
3. Should I_t = C_t/p or ((C_t+1)/p)-1? """
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
	rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]
	r[r < 0] += A.shape[1]
	column_indices = column_indices - r[:,np.newaxis]
	return A[rows, column_indices]

def FitSEIR(model,C_t,**kwargs):

	""" Organizational function that calls one of the functions below depending
	on what's provided at call. kwargs must be either 'rep_rate' OR 'R0'. """

	raise RuntimeError("FitSEIR and related methods are deprecated, use rainier.py instead.")

	## Check for clarity
	constraints = list(kwargs.keys())
	assert len(constraints) <= 1, "Provide either a reporting rate or an R0 value!"

	## Otherwise call the appropriate function
	if len(constraints) == 0:
		return _FitSEIRUnconstrained(model,C_t)
	elif constraints[0] == "rep_rate":
		print("Constraining model reporting rate in fitting.")
		return _FitSEIRWithRRConstraint(model,C_t,**kwargs)
	elif constraints[0] == "R0":
		print("Constraining model R0 in fitting.")
		return _FitSEIRWithR0Constraint(model,C_t,**kwargs)
	else:
		raise ValueError("{} is not a valid constraint.".format(constraints[0]))

## Internal fitting functions called by above
def _FitSEIRUnconstrained(model,C_t):

	""" Given an epi-curve, C_t, use scipy.minize to fit the model. """

	## Store the data and the length
	## of the fit period in time-steps.
	model.cases = C_t
	model.T_fit = len(C_t)

	## Find the optimal reporting rate
	result = minimize(lambda r: _neg_log_likelihood(r[0],C_t,model),
					  x0=np.array([0.1]),
					  method="L-BFGS-B",
					  bounds=[(1e-6,0.999)],
					  options={"maxiter":1e6})
	if not result["success"]:
		print("\nReporting rate inference failed!")
		print(result)
		#sys.exit()

	## Store the result
	model.inference_result = result
	model.p = result["x"][0]
	model.p_std = np.sqrt(result["hess_inv"].todense()[0,0])

	## Compute X_t, beta, and sig_eps given p
	## Construct estimates of S_t, E_t, and I_t
	approx_I = C_t/model.p
	approx_E = (model.D_e/model.D_i)*approx_I[model.D_e:]

	## Smooth the noise
	model.spline_e = SmoothingSpline(model.time[:len(approx_E)],approx_E,
									 lam=(model.D_e**4)/8)
	model.spline_i = SmoothingSpline(model.time[:len(approx_I)],approx_I,
									 lam=(model.D_i**4)/8)

	## Construct approximations
	smooth_I = model.spline_i(model.time)
	smooth_E = model.spline_e(model.time)
	new_exposures_t = smooth_E[1:] - (1.-(1./model.D_e))*smooth_E[:-1]
	smooth_S = model.S0 - np.cumsum(new_exposures_t)

	## Make the feature and response matrices for
	## least squares. In this case, since there's only the
	## lnbeta intercept to estimate, this is really simple.
	## We do this with warnings suppressed since log-issues are
	## handled via filtering below.
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		Y = np.log(smooth_E[1:] - (1.-(1./model.D_e))*smooth_E[:-1]) \
			- np.log(smooth_S) \
			- np.log(smooth_I[:-1]+model.z_t[:len(smooth_I)-1])

	## Clean up NaNs introduced by gaussian assumptions
	Y = Y[~np.isnan(Y)]

	## Compute least-squares estimates of lnbeta
	lnbeta = Y.mean()
	lnbeta_var = Y.var()/len(Y)

	## Compute transmission parameters
	model.beta = np.exp(lnbeta)
	model.beta_std = model.beta*np.sqrt(lnbeta_var)
	model.sig_eps = Y.std()

	## Finally compute R0
	model.R0 = model.beta*model.S0*model.D_i
	model.R0_std = model.beta_std*model.S0*model.D_i
	
	## Finish up
	model._fit = True
	return

def _FitSEIRWithR0Constraint(model,C_t,R0):

	""" Fit the SEIR model with R0 constrained to a particular value. """

	## Store the data and the length
	## of the fit period in time-steps.
	model.cases = C_t
	model.T_fit = len(C_t)

	## Incorporate the R0 constraint
	model.R0 = R0
	model.R0_std = np.nan

	## And the associated beta constraint
	model.beta = model.R0/(model.S0*model.D_i)
	model.beta_std = np.nan

	## Find the optimal reporting rate
	result = minimize(lambda r: _neg_log_likelihood_beta_constraint(r[0],C_t,model),
					  x0=np.array([0.1]),
					  method="L-BFGS-B",
					  bounds=[(1e-6,0.999)],
					  options={"maxiter":1e6})
	if not result["success"]:
		print("\nReporting rate inference failed!")
		print(result)
		#sys.exit()

	## Store the result
	model.inference_result = result
	model.p = result["x"][0]
	model.p_std = np.sqrt(result["hess_inv"].todense()[0,0])

	## Compute X_t, beta, and sig_eps given p
	## Construct estimates of S_t, E_t, and I_t
	approx_I = C_t/model.p
	approx_E = (model.D_e/model.D_i)*approx_I[model.D_e:]

	## Smooth the noise
	model.spline_e = SmoothingSpline(model.time[:len(approx_E)],approx_E,
									 lam=(model.D_e**4)/8)
	model.spline_i = SmoothingSpline(model.time[:len(approx_I)],approx_I,
									 lam=(model.D_i**4)/8)

	## Use the residual to estimate sig-eps
	model.sig_eps = np.sqrt(result["fun"]/model.T_fit)

	## Finish up
	model._fit = True
	return

def _FitSEIRWithRRConstraint(model,C_t,rep_rate):

	""" Fit the SEIR model with the reporting rate contrained to a particular value. """

	## Store the data and the length
	## of the fit period in time-steps.
	model.cases = C_t
	model.T_fit = len(C_t)

	## Store the reporting rate
	model.inference_result = "Reporting rate was constrained!"
	model.p = rep_rate
	model.p_std = np.nan

	## Compute X_t, beta, and sig_eps given p
	## Construct estimates of S_t, E_t, and I_t
	approx_I = C_t/model.p
	approx_E = (model.D_e/model.D_i)*approx_I[model.D_e:]

	## Smooth the noise
	model.spline_e = SmoothingSpline(model.time[:len(approx_E)],approx_E,
									 lam=(model.D_e**4)/8)
	model.spline_i = SmoothingSpline(model.time[:len(approx_I)],approx_I,
									 lam=(model.D_i**4)/8)

	## Construct approximations
	smooth_I = model.spline_i(model.time)
	smooth_E = model.spline_e(model.time)
	new_exposures_t = smooth_E[1:] - (1.-(1./model.D_e))*smooth_E[:-1]
	smooth_S = model.S0 - np.cumsum(new_exposures_t)

	## Make the feature and response matrices for
	## least squares. In this case, since there's only the
	## lnbeta intercept to estimate, this is really simple.
	## We do this with warnings suppressed since log-issues are
	## handled via filtering below.
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		Y = np.log(smooth_E[1:] - (1.-(1./model.D_e))*smooth_E[:-1]) \
			- np.log(smooth_S) \
			- np.log(smooth_I[:-1]+model.z_t[:len(smooth_I)-1])

	## Clean up NaNs introduced by gaussian assumptions
	Y = Y[~np.isnan(Y)]

	## Compute least-squares estimates of lnbeta
	lnbeta = Y.mean()
	lnbeta_var = Y.var()/len(Y)

	## Compute transmission parameters
	model.beta = np.exp(lnbeta)
	model.beta_std = model.beta*np.sqrt(lnbeta_var)
	model.sig_eps = Y.std()

	## Finally compute R0
	model.R0 = model.beta*model.S0*model.D_i
	model.R0_std = model.beta_std*model.S0*model.D_i
	
	## Finish up
	model._fit = True
	return

#### Internal functions
def _neg_log_likelihood(p,C_t,model):

	""" The log-posterior given the reporting rate, i.e. p(beta,alpha,X_t|p,C_t,M),
	which is the function we optimize w.r.t. p. """

	## Construct estimates of S_t, E_t, and I_t
	approx_I = C_t/p
	approx_E = (model.D_e/model.D_i)*approx_I[model.D_e:]

	## Smooth the noise
	spline_e = SmoothingSpline(model.time[:len(approx_E)],approx_E,
							   lam=(model.D_e**4)/8)
	spline_i = SmoothingSpline(model.time[:len(approx_I)],approx_I,
							   lam=(model.D_i**4)/8)

	## Construct approximations
	smooth_I = spline_i(model.time)
	smooth_E = spline_e(model.time)
	new_exposures_t = smooth_E[1:] - (1.-(1./model.D_e))*smooth_E[:-1]
	smooth_S = model.S0 - np.cumsum(new_exposures_t)

	## Make the feature and response matrices for
	## least squares. In this case, since there's only the
	## lnbeta intercept to estimate, this is really simple.
	## We do this with warnings suppressed since log-issues are
	## handled via filtering below.
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		Y = np.log(smooth_E[1:] - (1.-(1./model.D_e))*smooth_E[:-1]) \
			- np.log(smooth_S) \
			- np.log(smooth_I[:-1]+model.z_t[:len(smooth_I)-1])

	## Clean up NaNs introduced by gaussian assumptions
	Y = Y[~np.isnan(Y)]

	## Residual sum of squares is just the variance in
	## this case. 
	return Y.var()

def _neg_log_likelihood_beta_constraint(p,C_t,model):

	""" The log-posterior evaluated at a specific R0 value instead of determining
	the optimal via regression. """

	## Construct estimates of S_t, E_t, and I_t
	approx_I = C_t/p
	approx_E = (model.D_e/model.D_i)*approx_I[model.D_e:]

	## Smooth the noise
	spline_e = SmoothingSpline(model.time[:len(approx_E)],approx_E,
							   lam=(model.D_e**4)/8)
	spline_i = SmoothingSpline(model.time[:len(approx_I)],approx_I,
							   lam=(model.D_i**4)/8)

	## Construct approximations
	smooth_I = spline_i(model.time)
	smooth_E = spline_e(model.time)
	new_exposures_t = smooth_E[1:] - (1.-(1./model.D_e))*smooth_E[:-1]
	smooth_S = model.S0 - np.cumsum(new_exposures_t)

	## Make the feature and response matrices for
	## least squares. In this case, since there's only the
	## lnbeta intercept to estimate, this is really simple.
	## We do this with warnings suppressed since log-issues are
	## handled via filtering below.
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		Y = np.log(smooth_E[1:] - (1.-(1./model.D_e))*smooth_E[:-1]) \
			- np.log(smooth_S) \
			- np.log(smooth_I[:-1]+model.z_t[:len(smooth_I)-1])

	## Clean up NaNs introduced by gaussian assumptions
	Y = Y[~np.isnan(Y)]

	## Compare to what's expected based on the constraint
	rss = ((Y - np.log(model.beta))**2).sum()
	return rss