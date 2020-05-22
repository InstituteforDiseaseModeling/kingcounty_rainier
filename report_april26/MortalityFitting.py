""" MortalityFitting.py

Using non-linear least squares to fit importation pulse sizes to mortality 
data in the WDRS. """
import sys
from pathlib import Path
sys.path.append(Path("../"))

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(6)

## For time-series modeling and Reff estimation
from rainier.seir import LogNormalSEIR, SampleMortality
from rainier.rainier import continuous_time_posterior,\
							SplineTestingEpiCurve,\
							StepTestingEpiCurve
from rainier.stats import WeightedLeastSquares, ConstantWLS

## For feedback and progress throughout
from tqdm import tqdm

## For optimization
from scipy.optimize import minimize

## Helper functions
def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

def low_mid_high(samples):
	l0 = np.percentile(samples,1.,axis=0)
	h0 = np.percentile(samples,99.,axis=0)
	l1 = np.percentile(samples,2.5,axis=0)
	h1 = np.percentile(samples,97.5,axis=0)
	l2 = np.percentile(samples,25.,axis=0)
	h2 = np.percentile(samples,75.,axis=0)
	return l0, h0, l1, h1, l2, h2

def MeanMortality(model,beta_t,z_t,
				  mean_ifr=0.01,mean_delay=19):
	X = model.mean(beta_t,z_t)
	mean_exposures = X[1,1:] - (1. - (1./model.D_e))*X[1,:-1]
	destined_deaths = mean_ifr*mean_exposures
	mean_mortality = np.hstack([np.zeros((mean_delay+1,)),destined_deaths[:-mean_delay]])
	return mean_mortality

def RSS(theta,model,beta_t,pulse_indices,observed_deaths,
		mean_ifr=0.01,mean_delay=19):

	""" For use in the optimization approach. theta is from scipy.optimize, corresponding
	to pulse size at times based on pulse_indices. """

	## Set up z_t
	z_t = np.zeros((len(beta_t),))
	z_t[pulse_indices] = theta

	## Compute the mean
	model_mortality = MeanMortality(model,beta_t,z_t,mean_ifr,mean_delay)

	## Compute the residual
	residual = observed_deaths.values-model_mortality
	return np.sum(residual**2)

if __name__ == "__main__":

	## Get the county (or county group) specific dataset
	dataset = pd.read_pickle("../pickle_jar/aggregated_king_linelist_april26.pkl")

	## How do you handle data at the end, where increased testing and
	## lags might be an issue?
	dataset = dataset.loc[:"2020-04-20"]

	## Use the dataset to compute a testing-adjusted epicurve
	_version = 0
	if _version == 0:
		epi_curve = SplineTestingEpiCurve(dataset,debug=False)
	elif _version == 1:
		epi_curve = StepTestingEpiCurve(dataset,
										regimes=[pd.to_datetime("2020-03-02"),pd.to_datetime("2020-03-11")],
										debug=False)

	## Reindex cases to incorporate the 1-15-2020 importation.
	time = pd.date_range(start="01-15-2020",end=dataset.index[-1],freq="d")
	cases = dataset["cases"].reindex(time).fillna(0)
	deaths = dataset["deaths"].reindex(time).fillna(0)
	epi_curve = epi_curve.reindex(time).fillna(0)

	## Set up the transmission regression start
	tr_date = pd.to_datetime("2020-02-28")
	tr_start = (tr_date-time[0]).days

	## Set up importations
	importations = pd.Series(np.zeros(len(cases),),
							 index=cases.index,
							 name="importations")

	## Set up the initial susceptible population based on
	## the population of King County
	population = 2233163

	## Set up a model and fit it
	model = LogNormalSEIR(S0=population,
						  D_e=4,
						  D_i=8,
						  z_t=importations.values)

	## Fit the model with RAINIER using the testing corrected
	## epi-curve
	ps = np.linspace(0.01,1.,50)
	lnbeta = np.zeros((len(ps),len(model.time[tr_start:-5])))
	lnbeta_var = np.zeros((len(ps),len(model.time[tr_start:-5])))
	for t in enumerate(tqdm(ps)):
		i, p = t
		lnbeta[i], lnbeta_var[i] = continuous_time_posterior(p,model,epi_curve.values,
															 tr_start,debug=False)

	## Integrate (across the uniform components of the
	## p(p|C_t) and delta function approximated p(X_t|C_t,p))
	lnbeta = lnbeta.mean(axis=0)
	lnbeta_var = lnbeta_var.mean(axis=0)

	## Compute R0 point estimates
	R0_point_est = np.exp(lnbeta)*model.S0*model.D_i
	R0_point_est_std = np.exp(lnbeta)*np.sqrt(lnbeta_var)*model.S0*model.D_i
	r0_estimates = pd.DataFrame(np.array([R0_point_est,R0_point_est_std]).T,
								columns=["r0_t","std_err"],
								index=time[tr_start:tr_start+len(R0_point_est)])
	print("\nPoint estimates for R0:")
	print(r0_estimates)

	## Set sigma epsilon to variance contributions from the estimates in the
	## unabated tranmission period. Not sure if this is the right thing to do?
	sig_eps = pd.Series(np.sqrt(lnbeta_var),index=r0_estimates.index)
	sig_eps = sig_eps.reindex(time).fillna(method="ffill").fillna(method="bfill")
	model.sig_eps = sig_eps.values
	print("\nLog-normal mean-variance relationship:")
	print("sig_eps = {}".format(model.sig_eps))

	## Set up the beta_t scenario (forward filling with restriction)
	beta_t = r0_estimates["r0_t"]/model.S0/model.D_i
	cf_R0 = r0_estimates["r0_t"].loc[:"2020-03-01"].mean()
	cf_value = cf_R0/model.S0/model.D_i
	beta_t = beta_t.reindex(time).fillna(method="ffill").fillna(cf_value)

	## Now you have enough to compute the mean number of deaths in the
	## model conditional on z_t. Start by deciding where you want 
	## importation pulses.
	pulse_dates = [pd.to_datetime(d) for d in ["2020-01-15"]]#,"2020-02-19","2020-03-01"]]
	pulse_indices = [(d-time[0]).days for d in pulse_dates]

	## Then create a cost function and optimize it
	cost_function = lambda x: RSS(x,model,beta_t.values,pulse_indices,deaths)
	result = minimize(cost_function,
					  x0=5*np.ones((len(pulse_indices),)),
					  method="L-BFGS-B",
					  bounds=len(pulse_indices)*[(0,None)])
	print("\nImportation pulse optimization result:")
	print(result)

	## Summarize accordingly
	print("\nInferred pulses:")
	pulse_sizes = result["x"]
	pulse_std = np.sqrt(np.diag(result["hess_inv"].todense()))
	for d,s,std in zip(pulse_dates,pulse_sizes,pulse_std):
		print("{}: {} +/- {}".format(d.strftime("%m/%d/%Y"),s,std))

	## Set up the importations
	importations.loc[pulse_dates] = result["x"]
	model.z_t = importations.values

	## Sample the model
	samples = model.sample(beta_t.values)

	## For the samples, create samples of mortality
	#ifr_samples = np.random.uniform(low=0.002,high=0.036,size=(len(samples),))
	ifr_samples = np.random.gamma(shape=4,scale=0.25,size=(len(samples),))/100.
	delay_samples = np.exp(np.random.normal(2.8329,0.42,size=(len(samples),)))
	delay_samples = np.clip(delay_samples,None,samples.shape[2]-1)
	destined_deaths, deaths_occured = SampleMortality(model,samples,
													  ifr_samples,
													  np.round(delay_samples).astype(int))

	## How does that compare?
	fig, axes = plt.subplots(figsize=(18,7))
	axes_setup(axes)
	l0, h0, l1, h1, l2, h2 = low_mid_high(deaths_occured)
	axes.fill_between(cases.index[1:],l0,h0,color="#F98866",alpha=0.1)
	axes.fill_between(cases.index[1:],l1,h1,color="#F98866",alpha=0.3)
	axes.fill_between(cases.index[1:],l2,h2,color="#F98866",alpha=0.8,label="Transmission model fit to WDRS tests with ICs from mortality")
	axes.plot(time,MeanMortality(model,beta_t.values,importations.values),c="#F98866",lw=3)
	axes.plot(deaths.loc[deaths.loc[deaths!=0].index[0]:],
			  ls="None",color="k",marker="o",markersize=10,label="Daily King County COVID-19 deaths reported in WDRS")
	axes.legend(loc=2,frameon=False,fontsize=28)
	fig.tight_layout()
	fig.savefig(Path("../_plots/mortality.png"))

	## Finish up
	plt.show()
