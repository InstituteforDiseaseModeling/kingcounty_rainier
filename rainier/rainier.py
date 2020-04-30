""" rainier.py

Utilities for RAINIER (Reporting Adjusted Immuno-Naive, Infected, Exposed Regression). These
functions and classes manage analysis of case-data time series for comparison with regression
models. """
import sys
sys.path.append("..\\")
import warnings

## Standard imports
import numpy as np
import pandas as pd

## For debug/step-by-step plots
import matplotlib.pyplot as plt

## For estimating hidden state variables and
## associated uncertainty.
from rainier.splines import SmoothingSpline, SampleSpline

#### Data prep functions
def SplineTestingEpiCurve(dataset,debug=False):

	""" Create an epi curve based on fraction positive and smoothed total tests. dataset is a 
	dataframe with a daily time index with cases and negatives as columns. Smoothing here is done
	using a smoothing spline with a 3 day prior correlation. """

	## Compute fraction positive
	total_tests = dataset["cases"]+dataset["negatives"]
	fraction_positive = (dataset["cases"]/total_tests).fillna(0)

	## Compute spline smoothed total tests
	spline = SmoothingSpline(np.arange(len(total_tests)),total_tests.values,lam=((3**4)/8))
	smooth_tt = pd.Series(spline(np.arange(len(dataset))),
						  index=dataset.index) 
	smooth_tt.loc[smooth_tt<0] = 0

	## Compute the epicurve estimate
	epi_curve = fraction_positive*smooth_tt

	## Make a diagnostic plot if needed
	if debug:
		fig, axes = plt.subplots(3,1,sharex=True,figsize=(18,16))
		axes[0].plot(dataset["cases"],c="k",ls="dashed",lw=2,
					 label="WDRS COVID-19 positives for King County")
		axes[0].plot(fraction_positive*smooth_tt,c="xkcd:red wine",lw=3,
					 label="Epidemiological curve, based on smoothed tests,\nused to estimate " +r"$\beta(t)$")
		axes[1].plot(total_tests,c="k",ls="dashed",lw=2,
					 label="Raw total daily tests for King County")
		axes[1].plot(smooth_tt,c="xkcd:red wine",lw=3,
					 label="Smoothed tests with a 3 day correlation\ntime, correcting for testing fluctuations")
		axes[1].set_xlim(("2020-02-01",None))
		axes[2].plot(fraction_positive.loc[fraction_positive.loc[fraction_positive!=0].index[0]:],
					 c="grey",lw=3,label="Raw fraction positive, computed with\nWDRS positive and negative tests")
		for ax in axes:
			ax.legend(frameon=False,fontsize=28)
		axes[0].set_ylabel("Epi-curve")
		axes[1].set_ylabel("Total COVID-19 tests")
		axes[2].set_ylabel("Fraction positive")
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug.png")
		plt.show()
		sys.exit()

	return epi_curve

def StepTestingEpiCurve(dataset,regimes,debug=False):

	""" Create an epi curve based on fraction positive and step-wise total tests. dataset is a 
	dataframe with a daily time index with cases and negatives as columns. total tests is set to the mean
	in windows based on regimes, a list of date-times where splits are made. """

	## Compute fraction positive
	total_tests = dataset["cases"]+dataset["negatives"]
	fraction_positive = (dataset["cases"]/total_tests).fillna(0)

	## Step tests approximation
	regime_indices = [(d-dataset.index[0]).days for d in regimes]
	step_tt = np.split(total_tests,regime_indices)
	for i,s in enumerate(step_tt):
		if i == 0:
			continue
		s.loc[:] = int(np.round(s.mean()))
	step_tt = pd.concat(step_tt,axis=0)

	## Compute the epicurve estimate
	epi_curve = fraction_positive*step_tt

	## Make a diagnostic plot if needed
	if debug:
		fig, axes = plt.subplots(3,1,sharex=True,figsize=(18,16))
		axes[0].plot(dataset["cases"],c="k",ls="dashed",lw=2,label="Cases")
		axes[0].plot(epi_curve,c="xkcd:red wine",lw=2,label="Step-wise approximation")
		axes[1].plot(total_tests,c="k",lw=2,label="Total tests")
		axes[1].plot(step_tt,c="xkcd:red wine",lw=2,label="Step-function approximation")
		axes[1].set_xlim(("2020-02-01",None))
		axes[2].plot(fraction_positive,c="grey",lw=3,label="Raw fraction positive")
		for ax in axes:
			ax.legend(frameon=False,fontsize=28)
		axes[0].set_ylabel("Epi-curve")
		axes[1].set_ylabel("Total tests")
		axes[2].set_ylabel("Fraction positive from WDRS")
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug.png")
		plt.show()
		sys.exit()

	return epi_curve

#### Distribution functions
##############################################################################
def continuous_time_posterior(p,model,cases,tr_start,debug=False):

	""" This is the reporting conditional posterior on log_beta_t using a continuous time
	based approximation to the population in the exposed compartment. """


	## Start by constructing coarse estimates of I_t and E_t during the
	## testing period.
	coarse_I = cases/p
	coarse_E = (model.D_e/model.D_i)*coarse_I[model.D_e:]

	## Smooth the noise based on characteristic variation time
	## in each compartment.
	spline_e = SmoothingSpline(model.time[:len(model.time)-model.D_e],coarse_E,
							   lam=(model.D_e**4)/8)
	spline_i = SmoothingSpline(model.time,coarse_I,
							   lam=(model.D_i**4)/8)

	## Evaluate E and I plus covariance
	Ihat, Icov = spline_i(model.time,cov=True)
	Ehat, Ecov = spline_e(model.time,cov=True)

	## For the new exposures, we need a difference matrix (that should 
	## eventually be precomputed and stored in a RAINIER class?)
	e_diff_matrix = np.diag((-1.+(1./model.D_e))*np.ones((len(model.time)-1,)))\
				  + np.diag(np.ones((len(model.time)-2,)),k=1)
	e_diff_matrix = np.hstack([e_diff_matrix,np.zeros((len(model.time)-1,1))])
	e_diff_matrix[-1,-1] = 1
	delta_E = np.dot(e_diff_matrix,Ehat)
	delta_E_cov = np.dot(e_diff_matrix,np.dot(Ecov,e_diff_matrix.T))

	## Now, use a cumulative sum matrix (also to be precomputed in a 
	## RAINIER class and stored) to propogate uncertainty to s
	cum_sum = np.tril(np.ones((len(model.time)-1,len(model.time)-1)))
	Shat = model.S0 - np.dot(cum_sum,delta_E)
	Scov = np.dot(cum_sum,np.dot(delta_E_cov,cum_sum.T))

	## Finally, compute Yhat (the mean log transmission rate)
	Yhat = np.log(delta_E[tr_start:-model.D_e]) \
		   - np.log(Shat[tr_start-1:-model.D_e-1]) \
		   - np.log(Ihat[tr_start:-model.D_e-1]+model.z_t[tr_start:-model.D_e-1])

	## And approximate uncertainty - this is done without covariance
	## terms because the operating assumption of the model is a Markov
	## property (i.e. log(S_{t-1}) is applied to that point estimate only,
	## and the marginal distributions of the point estimates give you just
	## the diagonal elements)
	Yvar = np.diag(delta_E_cov)[tr_start:-model.D_e]/(delta_E[tr_start:-model.D_e]**2)\
		   + np.diag(Scov)[tr_start-1:-model.D_e-1]/(Shat[tr_start-1:-model.D_e-1]**2)\
		   + np.diag(Icov)[tr_start:-model.D_e-1]/(Ihat[tr_start:-model.D_e-1]**2)

	if debug:

		fig, axes = plt.subplots(4,1,sharex=True,figsize=(18,17))

		## Step 1: I_t
		I_std = np.sqrt(np.diag(Icov))
		axes[0].fill_between(model.time,Ihat-2.*I_std,Ihat+2.*I_std,color="#FFBB00",alpha=0.25)
		axes[0].plot(model.time,coarse_I,c="k",ls="dashed",label="Case data scaled by the reporting rate")
		axes[0].plot(model.time,Ihat,c="#FFBB00",label="Estimated infectious population")
		axes[0].axvline(tr_start,c="k")
		axes[0].axvline(len(model.time)-model.D_e-1,c="k")

		## Step 2: E_t
		E_std = np.sqrt(np.diag(Ecov))
		delta_E_std = np.sqrt(np.diag(delta_E_cov))
		axes[1].fill_between(model.time,Ehat-2.*E_std,Ehat+2.*E_std,color="#FB6542",alpha=0.25)
		axes[1].plot(model.time[:len(model.time)-model.D_e],coarse_E,c="grey",ls="dashed",
					 label="Scaled case data shifted by the latent period")
		axes[1].plot(model.time,Ehat,c="#FB6542",label="Estimated exposed population")
		axes[1].fill_between(model.time[:-1],delta_E-2.*delta_E_std,delta_E+2.*delta_E_std,color="k",alpha=0.25)
		axes[1].plot(model.time[:-1],delta_E,c="k",label="Estimated new exposures per day")
		axes[1].axvline(tr_start,c="k")
		axes[1].axvline(len(model.time)-model.D_e-1,c="k")

		## Step 3: S panel
		S_std = np.sqrt(np.diag(Scov))
		axes[2].fill_between(model.time[1:],Shat-2.*S_std,Shat+2.*S_std,color="#375E97",alpha=0.25)
		axes[2].plot(model.time[1:],Shat,color="#375E97",label="Estimated susceptible population\nbased on daily new exposures")
		axes[2].axvline(tr_start,c="k")
		axes[2].axvline(len(model.time)-model.D_e-1,c="k")

		## Step 4: log transmission rate panel
		Y_std = np.sqrt(Yvar)
		axes[3].fill_between(model.time[tr_start:-model.D_e-1],Yhat-2.*Y_std,Yhat+2.*Y_std,color="#3F681C",alpha=0.25)
		axes[3].plot(model.time[tr_start:-model.D_e-1],Yhat,c="#3F681C",
					 label="Estimated log transmission rate "+r"$(\log\beta)$"+"\ncomputed via hidden state estimates above")
		axes[3].axvline(tr_start,c="k")
		axes[3].axvline(len(model.time)-model.D_e-1,c="k")

		## Some labels
		for i, ax in enumerate(axes):
			ax.legend(frameon=False,fontsize=28)
			ax.set_yticks([])
			ax.set_ylabel("Step {}".format(i+1),fontweight="bold")
		axes[-1].set_xlabel("Model time step")

		## Annotations for the vertical bars
		axis_to_data = axes[3].transAxes + axes[3].transData.inverted()
		axes[3].text(tr_start,axis_to_data.transform((0,0.5))[1],
					 "Analysis period begins\n",fontsize=22,
					 horizontalalignment="center",verticalalignment="center",
					 rotation=90)
		axes[3].text(len(model.time)-model.D_e-1,axis_to_data.transform((0,0.5))[1],
					 "Analysis period ends\n",fontsize=22,
					 horizontalalignment="center",verticalalignment="center",
					 rotation=-90)
		## Done
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug.png")
		plt.show()
		sys.exit()

	return Yhat, Yvar

def continuous_time_posterior_sampler(p,model,cases,num_spline_samples,tr_start,debug=False):

	""" This is the reporting conditional posterior on log_beta_t using a continuous time
	based approximation to the population in the exposed compartment. Here, uncertainty is quantified
	via bootstrap samples. """

	## Start by constructing coarse estimates of I_t and E_t during the
	## testing period.
	coarse_I = cases/p
	coarse_E = (model.D_e/model.D_i)*coarse_I[model.D_e:]

	## Smooth the noise based on characteristic variation time
	## in each compartment.
	spline_e = SmoothingSpline(model.time[:len(model.time)-model.D_e],coarse_E,
							   lam=(model.D_e**4)/8)
	spline_i = SmoothingSpline(model.time,coarse_I,
							   lam=(model.D_i**4)/8)

	## Compute smoothed estimates
	smooth_I = SampleSpline(spline_i,model.time,num_samples=num_spline_samples)
	smooth_E = SampleSpline(spline_e,model.time,num_samples=num_spline_samples)
	new_exposures_t = smooth_E[:,1:] - (1.-(1./model.D_e))*smooth_E[:,:-1]
	smooth_S = model.S0 - np.cumsum(new_exposures_t,axis=0)

	## Create the regression response vector by comparing
	## S, E, and I estimates over time
	Y = np.log(smooth_E[:,tr_start+1:-model.D_e] - (1.-(1./model.D_e))*smooth_E[:,tr_start:-model.D_e-1]) \
		- np.log(smooth_S[:,tr_start-1:-model.D_e-1]) \
		- np.log(smooth_I[:,tr_start:-model.D_e-1]+model.z_t[np.newaxis,tr_start:-model.D_e-1])

	if debug:

		## S panel
		fig, axes = plt.subplots(4,1,sharex=True,figsize=(18,17))
		m = smooth_S.mean(axis=0)
		l = np.percentile(smooth_S,25.,axis=0)
		h = np.percentile(smooth_S,75.,axis=0)
		axes[2].fill_between(model.time[1:],l,h,color="#375E97",alpha=0.25)
		axes[2].plot(model.time[1:],m,color="#375E97",label="Estimated susceptible population\nbased on daily new exposures")
		axes[2].axvline(tr_start,c="k")
		axes[2].axvline(len(model.time)-model.D_e-1,c="k")

		## log transmission rate panel
		m = Y.mean(axis=0)
		l = np.percentile(Y,25.,axis=0)
		h = np.percentile(Y,75.,axis=0)
		axes[3].fill_between(model.time[tr_start:-model.D_e-1],l,h,color="#3F681C",alpha=0.25)
		axes[3].plot(model.time[tr_start:-model.D_e-1],m,c="#3F681C",
					 label="Estimated log transmission rate "+r"$(\log\beta)$"+"\ncomputed via hidden state estimates above")
		axes[3].axvline(tr_start,c="k")
		axes[3].axvline(len(model.time)-model.D_e-1,c="k")

		## I plot and E plot
		m_i = smooth_I.mean(axis=0)
		l_i = np.percentile(smooth_I,25.,axis=0)
		h_i = np.percentile(smooth_I,75.,axis=0)		
		m_e = smooth_E.mean(axis=0)
		l_e = np.percentile(smooth_E,25.,axis=0)
		h_e = np.percentile(smooth_E,75.,axis=0)	
		m_n = new_exposures_t.mean(axis=0)
		l_n = np.percentile(new_exposures_t,25.,axis=0)
		h_n = np.percentile(new_exposures_t,75.,axis=0)	
		axes[0].fill_between(model.time,l_i,h_i,color="#FFBB00",alpha=0.25)
		axes[0].plot(model.time,coarse_I,c="k",ls="dashed",label="Case data scaled by the reporting rate")
		axes[0].plot(model.time,m_i,c="#FFBB00",label="Estimated infectious population")
		axes[0].axvline(tr_start,c="k")
		axes[0].axvline(len(model.time)-model.D_e-1,c="k")
		axes[1].fill_between(model.time,l_e,h_e,color="#FB6542",alpha=0.25)
		axes[1].plot(model.time[:len(model.time)-model.D_e],coarse_E,c="grey",ls="dashed",
					 label="Scaled case data shifted by the latent period")
		axes[1].plot(model.time,m_e,c="#FB6542",label="Estimated exposed population")
		axes[1].fill_between(model.time[:-1],l_n,h_n,color="k",alpha=0.25)
		axes[1].plot(model.time[:-1],m_n,c="k",label="Estimated new exposures per day")
		axes[1].axvline(tr_start,c="k")
		axes[1].axvline(len(model.time)-model.D_e-1,c="k")

		## Some labels
		for i, ax in enumerate(axes):
			ax.legend(frameon=False,fontsize=28)
			ax.set_yticks([])
			ax.set_ylabel("Step {}".format(i+1),fontweight="bold")
		axes[-1].set_xlabel("Model time step")
		#axes[2].set(ylim=(model.S0*0.996,None))

		## Annotations for the vertical bars
		axis_to_data = axes[3].transAxes + axes[3].transData.inverted()
		axes[3].text(tr_start,axis_to_data.transform((0,0.5))[1],
					 "Analysis period begins\n",fontsize=22,
					 horizontalalignment="center",verticalalignment="center",
					 rotation=90)
		axes[3].text(len(model.time)-model.D_e-1,axis_to_data.transform((0,0.5))[1],
					 "Analysis period ends\n",fontsize=22,
					 horizontalalignment="center",verticalalignment="center",
					 rotation=-90)
		
		## Done
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug.png")
		plt.show()
		sys.exit()

	return Y