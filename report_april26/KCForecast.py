""" KCForecast.py

Scenario forecasting with the 4/26 version of the data and model. """
import sys
sys.path.append("..\\")

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

if __name__ == "__main__":

	## Get the county (or county group) specific dataset
	dataset = pd.read_pickle("..\\pickle_jar\\aggregated_king_linelist_april26.pkl")

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

	## Set up importations (value comes from mortality fitting 
	## script). 
	importations = pd.Series(np.zeros(len(cases),),
							 index=cases.index,
							 name="importations")
	importations.loc["01-15-2020"] = 39.125244097003204

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

	## Set sigma epsilon to variance contributions from the lnbeta
	## estimates, forward and backwards filling in time.
	sig_eps = pd.Series(np.sqrt(lnbeta_var),index=r0_estimates.index)
	model.sig_eps = sig_eps.values
	print("\nLog-normal mean-variance relationship:")
	print("sig_eps = {}".format(model.sig_eps))

	## Create beta_t scenarios, first one where SD continues
	forecast_time = pd.date_range(start=time[0],end="2020-06-15",freq="d")
	T = len(forecast_time)
	importations = importations.reindex(forecast_time).fillna(0)

	## Foward and back fill sig-eps
	sig_eps = sig_eps.reindex(forecast_time).fillna(method="ffill").fillna(method="bfill")

	## Create the baseline (as-is) scenario
	beta_t = r0_estimates["r0_t"]/model.S0/model.D_i
	cf_R0 = r0_estimates["r0_t"].loc[:"2020-03-01"].mean()
	cf_value = cf_R0/model.S0/model.D_i
	beta_t = beta_t.reindex(forecast_time).fillna(method="ffill").fillna(cf_value)

	## Create the declining Reff scenario
	decline = np.nan*beta_t
	decline.loc[:"05-01-2020"] = 1
	decline.loc["05-15-2020"] = 1.584
	decline = decline.interpolate()
	op_beta_t = decline*beta_t

	## Create the increasing Reff scenario
	increase = np.nan*beta_t
	increase.loc[:"05-01-2020"] = 1
	increase.loc["05-15-2020"] = 2.175
	increase = increase.interpolate()
	cf_beta_t = increase*beta_t

	## Compute the scenarios in terms of R0
	R0_scenario = beta_t*model.S0*model.D_i
	cf_R0_scenario = cf_beta_t*model.S0*model.D_i
	op_R0_scenario = op_beta_t*model.S0*model.D_i
	print("\nR0 scenario:")
	print(R0_scenario)

	## Sample the model
	samples = model.sample(beta_t.values,z_t=importations.values,sig_eps=sig_eps.values)
	cf_samples = model.sample(cf_beta_t.values,z_t=importations.values,sig_eps=sig_eps.values)
	op_samples = model.sample(op_beta_t.values,z_t=importations.values,sig_eps=sig_eps.values)

	## Use beta binomial over time to approximate reporting
	## rates
	i_samples = samples[:,2,tr_start:len(cases)]
	rr_samples = (cases.values[tr_start:]+1)/(i_samples+1)
	rr_t = rr_samples.mean(axis=0)
	rr_t_var = rr_samples.var(axis=0)

	## Fit a constant model to this data as well
	regime_change = pd.to_datetime("2020-03-09")
	X = np.array([(time <= regime_change).astype(np.float64),
				  (time > regime_change).astype(np.float64),
				  (time.weekday.isin({5,6})).astype(np.float64)]).T
	p, p_var, _ = WeightedLeastSquares(X[tr_start:],rr_t,
									   weights=1./rr_t_var,
									   standardize=False)
	rr = np.dot(X,p)
	rr_std = np.sqrt(np.diag(np.dot(X,np.dot(p_var,X.T))))
	X_output = np.array([[1,0,0],
						 [0,1,0],
						 [0,1,1]])
	rr_o = np.dot(X_output,p)
	rr_o_std = np.sqrt(np.diag(np.dot(X_output,np.dot(p_var,X_output.T))))
	print("Reporting rate estimate before {} = {} +/- {}".format(regime_change,rr_o[0],rr_o_std[0]))
	print("Reporting rate estimate after {} = {} +/- {}".format(regime_change,rr_o[1],rr_o_std[1]))
	print("Reporting rate estimate after {} on weekends = {} +/- {}".format(regime_change,rr_o[2],rr_o_std[2]))

	## Sample the model for cases, first by forward filling the reporting rate
	## if needed.
	rr = pd.Series(rr,index=cases.index).reindex(forecast_time).fillna(method="ffill")
	mask = (rr.index > time[-1]) & (rr.index.weekday.isin({5,6}))
	#rr.loc[mask] = rr.loc[mask] + p[-1]
	rr = rr.values

	## Sample the model for cases
	case_samples = np.random.binomial(np.round(samples[:,2,:]).astype(int),
									  p=rr)
	cf_case_samples = np.random.binomial(np.round(cf_samples[:,2,:]).astype(int),
										 p=rr)
	op_case_samples = np.random.binomial(np.round(op_samples[:,2,:]).astype(int),
										 p=rr)
	
	## Plot the # of infections over time
	fig, axes = plt.subplots(figsize=(22,10))
	axes_setup(axes)

	## Top panel
	sd_color = "#2C7873"
	op_color = "#F98866"
	cf_color = "grey"
	cf_l0, cf_h0, cf_l1, cf_h1, cf_l2, cf_h2 = low_mid_high(cf_case_samples)
	op_l0, op_h0, op_l1, op_h1, op_l2, op_h2 = low_mid_high(op_case_samples)
	sd_l0, sd_h0, sd_l1, sd_h1, sd_l2, sd_h2 = low_mid_high(case_samples)
	axes.fill_between(forecast_time,sd_l2,sd_h2,color=sd_color,alpha=0.25)
	ylim = axes.get_ylim()
	axes.fill_between(forecast_time,cf_l2,cf_h2,color=cf_color,alpha=0.25)
	axes.fill_between(forecast_time,op_l2,op_h2,color=op_color,alpha=0.25)
	axes.plot(forecast_time,cf_case_samples.mean(axis=0),lw=3,c=cf_color,label=r"Expectation with R$_e$ gradually increasing to mid-March levels (starting May 1)")
	axes.plot(forecast_time,op_case_samples.mean(axis=0),lw=3,c=op_color,label=r"Expectation with R$_e$ gradually increasing to late-March levels (starting May 1)")
	axes.plot(forecast_time,case_samples.mean(axis=0),lw=3,color=sd_color,label=r"Expectation with R$_e$ maintained at current levels")
	axes.plot(cases.loc[tr_date:],
			  ls="None",marker="o",markersize=10,
			  markeredgecolor="k",markerfacecolor="k",markeredgewidth=2,
			  label="WADoH daily positive COVID-19 tests in King County used for model fitting")

	## Details
	axes.set_ylim((ylim[0],375))
	axes.set_ylabel("Daily COVID-19 positives")
	axes.set_xticks(forecast_time[::20])
	axes.legend(loc=2,frameon=False,fontsize=28)
	fig.tight_layout()

	## Add an inset for the timeseries
	inset_dim = [0.115, 0.31, 0.28, 0.31] ## left, bottom, width, height
	axes2 = fig.add_axes(inset_dim)
	axes2.grid(color="grey",alpha=0.2)

	## Make a plot of night time lights over time
	axes2.plot(cf_R0_scenario.loc["2020-03-01":],color=cf_color,lw=3)
	axes2.plot(op_R0_scenario.loc["2020-03-01":],color=op_color,lw=3)
	axes2.plot(R0_scenario.loc["2020-03-01":],color=sd_color,lw=3)
	axes2.set(ylabel=r"R$_{e}(t)$")
	axes2.set_xticks([pd.to_datetime("2020-03-01"),
					  pd.to_datetime("2020-04-01"),
					  pd.to_datetime("2020-05-01"),
					  pd.to_datetime("2020-06-01")])
	axes2.set_xticklabels(["March 1","April 1","May 1","June 1"])
	fig.savefig("..\\_plots\\forecast.png")

	plt.show()