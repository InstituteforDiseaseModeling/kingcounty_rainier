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