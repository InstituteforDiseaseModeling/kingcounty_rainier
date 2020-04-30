""" splines.py

Smoothing splines and related functions """
import sys

## Standard imports 
import numpy as np
import pandas as pd

## For the B-spline basis
from scipy.interpolate import BSpline

#### Smoothing spline
##############################################################################
class SmoothingSpline(object):

	""" Main cubic smoothing spline class. """

	def __init__(self,x,y,lam=10.,max_knots=250,order=3):

		""" Initialize and fit the cubic smoothing spline, using the Bspline basis from 
		scipy. 
		
		x: A numpy array of univariate independent variables.
		y: A numpy array of univariate response variables.
		lam: smoothing parameter
		max_knots: max number of knots (i.e. max number of spline basis functions)."""

		## Start by storing the data
		assert len(x) == len(y), "Independent and response variable vectors must have the same length."
		self.x = x
		self.y = y
		self.order = 3
		self.lam = lam

		## Then compute the knots, refactoring to evenly
		## spaced knots if the number of unique values is too
		## large.
		self.knots = np.sort(np.unique(self.x))
		if len(self.knots) > max_knots:
			self.knots = np.linspace(self.knots[0],self.knots[-1],max_knots)

		## Construct the spline basis given the knots
		self._aug_knots = np.hstack([self.knots[0]*np.ones((order,)),self.knots,self.knots[-1]*np.ones((order,))])
		self.splines = [BSpline(self._aug_knots,coeffs,order) for coeffs in np.eye(len(self.knots)+order-1)]

		## Finally, with the basis functions, we can fit the 
		## smoother.
		self._fit()

	def _fit(self):

		""" Subroutine for fitting, called by init. """

		## Start the fit procedure by constructing the matrix B
		B = np.array([sp(self.x) for sp in self.splines]).T

		## Then, construct the penalty matrix by first computing second derivatives
		## on the knots and then approximating the integral of second derivative products
		## with the trapezoid rule.
		d2B = np.array([sp.derivative(2)(self.knots) for sp in self.splines])
		weights = np.ones(self.knots.shape)
		weights[1:-1] = 2.
		Omega = np.dot(d2B,np.dot(np.diag(weights),d2B.T))

		## Finally, invert the matrices to construct values of interest.
		self.ridge_op = np.linalg.inv(np.dot(B.T,B)+self.lam*Omega)

		## From there, we can compute the coefficient matrix H and the
		## S matrix (i.e. the hat matrix).
		H = np.dot(self.ridge_op,B.T)
		self.S = np.dot(B,H)
		self.edof = np.trace(self.S)
		self.gamma = np.dot(H,self.y)

		## Finally, construct the smoothing spline
		self.smoother = BSpline(self._aug_knots,self.gamma,self.order)

		## And compute the covariance matrix of the coefficients
		y_hat = self.smoother(self.x)
		self.rss = np.sum((self.y-y_hat)**2)
		self.var = self.rss/(len(self.y)-self.edof)
		self.cov = self.var*self.ridge_op
		
		return

	def __call__(self,x,cov=False):

		""" Evaluate it """

		## if you want the covariance matrix
		if cov:

			## Set up the matrix of splines evaluations and similarity transform the
			## covariance in the coefficients
			F = np.array([BSpline(self._aug_knots,g,self.order)(x)\
						  for g in np.eye(len(self.knots)+self.order-1)]).T
			covariance_matrix = np.dot(F,np.dot(self.cov,F.T))

			## Evaluate the mean using the point estimate
			## of the coefficients
			point_estimate = np.dot(F,self.gamma)

			return point_estimate, covariance_matrix

		else:
			return self.smoother(x)

	def derivative(self,x,degree=1):

		""" Evaluate the derivative of degree = degree. """

		return self.smoother.derivative(degree)(x)

	def correlation_time(self):

		""" Use the inferred covariance matrix to compute and estimate of the correlation time
		by approximating the width of correlation for a central knot with it's neighbors. """

		## Select a central knot
		i_mid = int(len(self.knots)/2)
		
		## Compute a normalized distribution
		distribution = np.abs(self.cov[i_mid][1:-1])
		distribution = distribution/trapz(distribution,x=self.knots)

		## Compute the mean and variance
		avg = self.knots[i_mid-1]
		var = trapz(distribution*(self.knots**2),x=self.knots) - avg**2

		return np.sqrt(var)

def BootstrapCI(smoother,time,low_p=2.5,high_p=97.5,num_samples=5000):

	""" Compute bootstrap confidence intervals using a smoother over numpy array
	time. Smoother must be an instance of the class above. """

	## Sample the spline coefficients
	gamma_samples = np.random.multivariate_normal(mean=smoother.gamma,
												  cov=smoother.cov,
												  size=(num_samples,))

	## Compute smoother
	smooth_samples = np.array([BSpline(smoother._aug_knots,g,smoother.order)(time) for g in gamma_samples])

	## Summarize
	low = np.percentile(smooth_samples,low_p,axis=0)
	high = np.percentile(smooth_samples,high_p,axis=0)

	return smooth_samples, low, high

def SampleSpline(smoother,time,num_samples=5000):

	""" Compute samples from the fitted smoother. """

	## Sample the spline coefficients
	gamma_samples = np.random.multivariate_normal(mean=smoother.gamma,
												  cov=smoother.cov,
												  size=(num_samples,))

	## Compute smoother
	smooth_samples = np.array([BSpline(smoother._aug_knots,g,smoother.order)(time) for g in gamma_samples])

	return smooth_samples	

if __name__ == "__main__":

	## For testing
	import matplotlib.pyplot as plt

	## Get some hypothetical data
	dataset = pd.read_pickle("..\\pickle_jar\\wa_doh_timeseries.pkl")
	dataset = dataset.loc["king"].groupby("time").sum()
	dataset = dataset.iloc[:-1]

	## Select a test
	## 0: bootstrap vs multivariate normal CI estimates
	## 1: constructing E_t, I_t, and S_t with bootstrap and MVN
	_test = 1

	if _test == 0:
	
		## Get case data for smoothing
		time = np.arange(len(dataset))
		extrapolated_time = np.arange(len(dataset)+5)
		cases = dataset["cases"].values

		## Compute a spline smooth
		spline = SmoothingSpline(time,cases,lam=8**3)

		## Compute the mean
		point_est = spline(time)

		## Compute some intervals with a bootstrap
		samples, low, high = BootstrapCI(spline,time,low_p=2.5,high_p=97.5,num_samples=5000)

		## Compute the same via properties of multivariate gaussians
		x_mean, x_cov = spline(extrapolated_time,cov=True)
		x_std = np.sqrt(np.diag(x_cov))

		## A comparison plot
		fig, axes = plt.subplots(figsize=(18,9))
		axes.plot(time,cases,ls="dashed",color="k",lw=2)

		## Plot the bootstrap CI
		axes.fill_between(time,low,high,color="grey",alpha=0.25)

		## Plot the MVN CI
		axes.plot(extrapolated_time,x_mean-2.*x_std,c="xkcd:red wine",ls="dashed")
		axes.plot(extrapolated_time,x_mean+2.*x_std,c="xkcd:red wine",ls="dashed")

		## Plot the mean
		axes.plot(time,point_est,color="grey")
		axes.plot(extrapolated_time,x_mean,color="xkcd:red wine",ls="dashed")

		## Finish up
		fig.tight_layout()
		plt.show()

	elif _test == 1:

		## Get case data for smoothing and set the 
		## reporting rate
		time = np.arange(len(dataset))
		cases = dataset["cases"].values
		tr_start = 44

		## Set the 'disease' parameters
		D_e = 4
		D_i = 8
		p = 0.1
		S0 = 2e6

		## Compute coarse_I and coarse_E
		coarse_I = cases/p
		coarse_E = (D_e/D_i)*coarse_I[D_e:]

		## Smooth the noise based on characteristic variation time
		## in each compartment.
		spline_e = SmoothingSpline(time[:len(time)-D_e],coarse_E,
							   	   lam=(D_e**4)/8)
		spline_i = SmoothingSpline(time,coarse_I,
								   lam=(D_i**4)/8)

		## Compute smoothed estimates via sampling
		num_spline_samples = 5000
		smooth_I = SampleSpline(spline_i,time,num_samples=num_spline_samples)
		smooth_E = SampleSpline(spline_e,time,num_samples=num_spline_samples)
		new_exposures_t = smooth_E[:,1:] - (1.-(1./D_e))*smooth_E[:,:-1]
		smooth_S = S0 - np.cumsum(new_exposures_t,axis=0)

		## Create the regression response vector by comparing
		## S, E, and I estimates over time
		Y = np.log(smooth_E[:,tr_start+1:-D_e] - (1.-(1./D_e))*smooth_E[:,tr_start:-D_e-1]) \
			- np.log(smooth_S[:,tr_start-1:-D_e-1]) \
			- np.log(smooth_I[:,tr_start:-D_e-1])

		## Now, do the same but propogate uncertainty analytically
		Ihat, Icov = spline_i(time,cov=True)
		Ehat, Ecov = spline_e(time,cov=True)

		## For the new exposures, we need a difference matrix (that should 
		## eventually be precomputed and stored in a RAINIER class?)
		e_diff_matrix = np.diag((-1.+(1./D_e))*np.ones((len(time)-1,)))\
					  + np.diag(np.ones((len(time)-2,)),k=1)
		e_diff_matrix = np.hstack([e_diff_matrix,np.zeros((len(time)-1,1))])
		e_diff_matrix[-1,-1] = 1
		delta_E = np.dot(e_diff_matrix,Ehat)
		delta_E_cov = np.dot(e_diff_matrix,np.dot(Ecov,e_diff_matrix.T))
		delta_E_std = np.sqrt(np.diag(delta_E_cov))

		## Now, use a cumulative sum matrix (also to be precomputed in a 
		## RAINIER class and stored) to propogate uncertainty to s
		cum_sum = np.tril(np.ones((len(time)-1,len(time)-1)))
		Shat = S0 - np.dot(cum_sum,delta_E)
		Scov = np.dot(cum_sum,np.dot(delta_E_cov,cum_sum.T))
		S_std = np.sqrt(np.diag(Scov))

		## Finally, compute Yhat (the mean log attack rate)
		a_start = tr_start
		Yhat = np.log(delta_E[a_start:-D_e]) \
			   - np.log(Shat[a_start-1:-D_e-1]) \
			   - np.log(Ihat[a_start:-D_e-1])

		## And approximate uncertainty - this is done without covariance
		## terms because the operating assumption of the model is a Markov
		## property (i.e. log(S_{t-1}) is applied to that point estimate only,
		## and the marginal distributions of the point estimates give you just
		## the diagonal elements)
		Yvar = np.diag(delta_E_cov)[a_start:-D_e]/(delta_E[a_start:-D_e]**2)\
			   + np.diag(Scov)[a_start-1:-D_e-1]/(Shat[a_start-1:-D_e-1]**2)\
			   + np.diag(Icov)[a_start:-D_e-1]/(Ihat[a_start:-D_e-1]**2)
		Ystd = np.sqrt(Yvar)

		## make a figure for comparisons
		fig, axes = plt.subplots(4,1,sharex=True,figsize=(18,17))
		m = smooth_S.mean(axis=0)
		l = np.percentile(smooth_S,2.5,axis=0)
		h = np.percentile(smooth_S,97.5,axis=0)
		axes[2].fill_between(time[1:],l,h,color="#375E97",alpha=0.25)
		axes[2].plot(time[1:],m,color="#375E97",label="Estimated susceptible population\nbased on daily new exposures")
		axes[2].plot(time[1:],Shat,color="C1",ls="dashed")
		axes[2].plot(time[1:],Shat-2.*S_std,color="C1",ls="dashed")
		axes[2].plot(time[1:],Shat+2.*S_std,color="C1",ls="dashed")
		axes[2].axvline(tr_start,c="k")
		axes[2].axvline(len(time)-D_e-1,c="k")

		## log attack rate panel
		m = Y.mean(axis=0)
		l = np.percentile(Y,2.5,axis=0)
		h = np.percentile(Y,97.5,axis=0)
		axes[3].fill_between(time[tr_start:-D_e-1],l,h,color="#3F681C",alpha=0.25)
		axes[3].plot(time[tr_start:-D_e-1],m,c="#3F681C",
					 label="Estimated log attack rate "+r"$(\log\beta)$"+"\ncomputed via hidden state estimates above")
		axes[3].plot(time[a_start:-D_e-1],Yhat,c="C1",ls="dashed")
		axes[3].plot(time[a_start:-D_e-1],Yhat-2.*Ystd,c="C1",ls="dashed")
		axes[3].plot(time[a_start:-D_e-1],Yhat+2.*Ystd,c="C1",ls="dashed")
		axes[3].axvline(tr_start,c="k")
		axes[3].axvline(len(time)-D_e-1,c="k")

		## I plot and E plot
		m_i = smooth_I.mean(axis=0)
		l_i = np.percentile(smooth_I,2.5,axis=0)
		h_i = np.percentile(smooth_I,97.5,axis=0)		
		m_e = smooth_E.mean(axis=0)
		l_e = np.percentile(smooth_E,2.5,axis=0)
		h_e = np.percentile(smooth_E,97.5,axis=0)	
		m_n = new_exposures_t.mean(axis=0)
		l_n = np.percentile(new_exposures_t,2.5,axis=0)
		h_n = np.percentile(new_exposures_t,97.5,axis=0)	
		axes[0].fill_between(time,l_i,h_i,color="#FFBB00",alpha=0.25)
		axes[0].plot(time,coarse_I,c="k",ls="dashed",label="Case data scaled by the reporting rate")
		axes[0].plot(time,m_i,c="#FFBB00",label="Estimated infectious population")
		axes[0].axvline(tr_start,c="k")
		axes[0].axvline(len(time)-D_e-1,c="k")
		axes[1].fill_between(time,l_e,h_e,color="#FB6542",alpha=0.25)
		axes[1].plot(time[:len(time)-D_e],coarse_E,c="grey",ls="dashed",
					 label="Scaled case data shifted by the latent period")
		axes[1].plot(time,m_e,c="#FB6542",label="Estimated exposed population")
		axes[1].fill_between(time[:-1],l_n,h_n,color="k",alpha=0.25)
		axes[1].plot(time[:-1],m_n,c="k",label="Estimated new exposures per day")
		axes[1].plot(time[:-1],delta_E,c="C1",ls="dashed")
		axes[1].plot(time[:-1],delta_E-2.*delta_E_std,c="C1",ls="dashed")
		axes[1].plot(time[:-1],delta_E+2.*delta_E_std,c="C1",ls="dashed")
		axes[1].axvline(tr_start,c="k")
		axes[1].axvline(len(time)-D_e-1,c="k")

		## Some labels
		for i, ax in enumerate(axes):
			ax.legend(frameon=False,fontsize=18)
			ax.set_yticks([])

		## Annotations for the vertical bars
		axis_to_data = axes[3].transAxes + axes[3].transData.inverted()
		axes[3].text(tr_start,axis_to_data.transform((0,0.5))[1],
					 "Analysis period begins\n",fontsize=18,
					 horizontalalignment="center",verticalalignment="center",
					 rotation=90)
		axes[3].text(len(time)-D_e-1,axis_to_data.transform((0,0.5))[1],
					 "Analysis period ends\n",fontsize=18,
					 horizontalalignment="center",verticalalignment="center",
					 rotation=-90)
		
		## Done
		fig.tight_layout()
		plt.show()
