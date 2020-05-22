""" PanelRAINIER.py

County level estimates of Reff over time, with comparison of different covariate driven
models. Same as RAINIER.py but the plots are adjusted to be multi-panel instead of individual. """
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

## For histograms
from scipy.stats import gaussian_kde

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

def GetSCANPrevalence(version="recent"):

	if version == "recent":
		raise NotImplementedError("This version doesn't have unpublished results!")

	elif version == "published":
		scan_result = [("2020-03-23","2020-03-29",0.32,0.08,1.18),
					   ("2020-03-29","2020-04-04",0.27,0.07,0.95),
					   ("2020-04-04","2020-04-10",0.07,0.01,0.36)]

	return scan_result

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

	## Plot the results
	fig, axes = plt.subplots(figsize=(18,9))
	axes_setup(axes)

	## Epi data
	label = r"R$_{e}$ estimates for King County based on WDRS test data"
	axes.errorbar(r0_estimates.index+pd.to_timedelta(0.1,unit="d"),R0_point_est,yerr=2.*R0_point_est_std,
				  color="k",marker="o",markersize=10,ls="None",label=label,zorder=10)

	## Annotation
	axes.set(ylim=(0.,4))
	axis_to_data = axes.transAxes + axes.transData.inverted()
	policy_date = pd.to_datetime("2020-03-24")-pd.to_timedelta(0.1,unit="d")
	axes.axhline(1.0,c="#4D648D",lw=2,ls="dashed")#,label=r"Threshold for declining transmission, R$_{e}=1$")
	axes.text(axis_to_data.transform((0.,0.))[0],0.95,
			  r"Threshold for declining transmission, R$_{e}=1$",
			  horizontalalignment="left",verticalalignment="top",
			  fontsize=32,color="#4D648D")
	axes.legend(loc=1,frameon=False,fontsize=28)
	axes.set_ylabel(r"Effective reproductive number (R$_{e}$)")
	axes.set_xticks(r0_estimates.index[::6])
	fig.tight_layout()
	fig.savefig("../_plots/r0.png")

	## Set up a forecast time for plotting
	plot_time = pd.date_range(start=time[0],end="2020-04-29",freq="d")
	T = len(plot_time)
	importations = importations.reindex(plot_time).fillna(0)

	## Set sigma epsilon to variance contributions from the lnbeta
	## estimates, forward and backwards filling in time.
	sig_eps = pd.Series(np.sqrt(lnbeta_var),index=r0_estimates.index)
	sig_eps = sig_eps.reindex(plot_time).fillna(method="ffill").fillna(method="bfill")
	model.sig_eps = sig_eps.values
	print("\nLog-normal mean-variance relationship:")
	print("sig_eps = {}".format(model.sig_eps))

	## Set up the beta_t scenario (forward filling with restriction)
	beta_t = r0_estimates["r0_t"]/model.S0/model.D_i
	cf_R0 = r0_estimates["r0_t"].loc[:"2020-03-01"].mean()
	cf_value = cf_R0/model.S0/model.D_i
	beta_t = beta_t.reindex(plot_time).fillna(method="ffill").fillna(cf_value)

	## Sample the model
	samples = model.sample(beta_t.values,z_t=importations.values)

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

	## SET UP A BIG FIGURE FOR MODEL FITTING
	fig, axes = plt.subplots(3,1,figsize=(18,21))
	for ax in axes:
		axes_setup(ax)

	## Plot RR samples over time
	axes[1].errorbar(time[tr_start:],rr_t,yerr=2.*np.sqrt(rr_t_var),
					 color="k",marker="o",markersize=12,ls="None",
					 label=r"Daily estimates based on comparing WDRS positives to infections in the model")
	axes[1].fill_between(time[tr_start:],rr[tr_start:]-2.*rr_std[tr_start:],rr[tr_start:]+2.*rr_std[tr_start:],
					  	 facecolor="#FFBB00",edgecolor="None",alpha=0.3)	
	axes[1].plot(time[tr_start:],rr[tr_start:],color="#FFBB00",
				 label="Aggregated estimates in 2 reporting periods,\npre- and post-March 10, accounting for weekends")	
	axes[1].set_xticks(time[tr_start:][::5])
	axes[1].legend(loc=2,frameon=False,fontsize=28)
	axes[1].set(ylim=(0,0.12))
	axes[1].set_ylabel("Daily probability an infection is reported")
	axes[1].set_xticks(time[tr_start:][::6])

	## Sample the model for cases, first by forward filling the reporting rate
	## if needed.
	rr = pd.Series(rr,index=cases.index).reindex(plot_time).fillna(method="ffill")
	mask = (rr.index > time[-1]) & (rr.index.weekday.isin({5,6}))
	rr.loc[mask] = rr.loc[mask] + p[-1]
	rr = rr.values
	case_samples = np.random.binomial(np.round(samples[:,2,:]).astype(int),
									  p=rr)
	
	## Plot the # of infections over time (v2)
	sd_l0, sd_h0, sd_l1, sd_h1, sd_l2, sd_h2 = low_mid_high(case_samples)
	axes[2].fill_between(plot_time,sd_l0,sd_h0,color="#2C7873",alpha=0.1,zorder=1)
	axes[2].fill_between(plot_time,sd_l1,sd_h1,color="#2C7873",alpha=0.2,zorder=2)
	axes[2].fill_between(plot_time,sd_l2,sd_h2,color="#2C7873",alpha=0.8,zorder=3,label="Transmission model fit to WDRS tests and mortality")
	axes[2].plot(plot_time,case_samples.mean(axis=0),lw=3,color="#2C7873")
	axes[2].plot(cases.loc["2020-02-26":],
			  ls="None",marker="o",markersize=10,
			  markeredgecolor="k",markerfacecolor="k",markeredgewidth=2,zorder=4,
			  label="Daily WDRS positive COVID-19 tests in King County")
	axes[2].plot(importations.loc[importations != 0],
			  ls="None",marker="o",markersize=12,
			  markeredgecolor="xkcd:red wine",markerfacecolor="None",markeredgewidth=2,
			  label="COVID-19 importations on January 15")

	## Details
	axes[2].legend(loc=2,frameon=False,fontsize=28)
	axes[2].set_ylabel("COVID-19 cases")
	
	## For the samples, create samples of mortality
	ifr_samples = np.random.gamma(shape=4,scale=0.25,size=(len(samples),))/100.
	delay_samples = np.exp(np.random.normal(2.8329,0.42,size=(len(samples),)))
	delay_samples = np.clip(delay_samples,None,samples.shape[2]-1)
	destined_deaths, deaths_occured = SampleMortality(model,samples,
													  ifr_samples,
													  np.round(delay_samples).astype(int))

	## How does that compare?
	l0, h0, l1, h1, l2, h2 = low_mid_high(deaths_occured)
	axes[0].fill_between(plot_time[1:],l0,h0,color="#F98866",alpha=0.1)
	axes[0].fill_between(plot_time[1:],l1,h1,color="#F98866",alpha=0.3)
	axes[0].fill_between(plot_time[1:],l2,h2,color="#F98866",alpha=0.6,label="Transmission model fit to WDRS tests and mortality")
	axes[0].plot(plot_time[1:],deaths_occured.mean(axis=0),lw=3,color="#F98866")
	axes[0].plot(dataset.loc[dataset.loc[dataset["deaths"]!=0].index[0]:,"deaths"],
			  ls="None",color="k",marker="o",markersize=10,label="Daily COVID-19 deaths in King County reported in WDRS")
	axes[0].legend(loc=2,frameon=False,fontsize=28)
	axes[0].set_ylabel("COVID-19 deaths")
	
	## Save the multipanel
	fig.tight_layout()
	fig.savefig("../_plots/fit.png")

	## Compute active infections (and describe)
	prevalence = pd.DataFrame((samples[:,1,:] + samples[:,2,:]).T/population,
							   index=plot_time).T
	print("\nPrevalence:")
	print(prevalence[time[-1]].describe(percentiles=[0.025,0.25,0.5,0.75,0.975]))
	attack_rate = pd.DataFrame((model.S0 - samples[:,0,:]).T/population,
							   index=plot_time).T
	print("\nAttack rate:")
	print(attack_rate[time[-1]].describe(percentiles=[0.025,0.25,0.5,0.75,0.975]))

	## And the cumulative reporting rate
	total_cases = dataset["cases"].sum()
	cum_rr_samples = 100*total_cases/attack_rate[time[-1]]/population
	print("\nCumulative reporting rate:")
	cum_rr = cum_rr_samples.describe(percentiles=[0.025,0.25,0.5,0.75,0.975]) 
	print(cum_rr)

	## Make a joint prevalence incidence figure
	fig, axes = plt.subplots(2,1,figsize=(18,14))
	for ax in axes:
		axes_setup(ax)

	## Plot prevalence over time
	scan_result = GetSCANPrevalence("published")
	l0, h0, l1, h1, l2, h2 = low_mid_high(100*prevalence.values)
	axes[0].fill_between(plot_time,l0,h0,color="#7F152E",alpha=0.1)
	axes[0].fill_between(plot_time,l1,h1,color="#7F152E",alpha=0.3)
	axes[0].fill_between(plot_time,l2,h2,color="#7F152E",alpha=0.8,label="Estimated prevalence using\nthe transmission model")
	axes[0].plot(plot_time,(100*prevalence.values).mean(axis=0),color="#7F152E",lw=3)
	for t in scan_result:
		d1, d2, m, l, h = t
		d1, d2 = pd.to_datetime(d1), pd.to_datetime(d2)
		axes[0].fill_between([d1,d2],[l,l],[h,h],alpha=0.4,facecolor="grey",edgecolor="None")
		axes[0].plot([d1,d2],[m,m],lw=3,color="k")
	axes[0].plot([],c="k",lw=3,label="Prevalence estimates from SCAN")
	axes[0].legend(loc=2,frameon=False,fontsize=28)
	axes[0].set_ylabel(r"COVID-19 prevalence (%)")
	axes[0].set_xlim(("2020-02-01",None))

	## Plot cummulative incidence/attack rate over time
	l0, h0, l1, h1, l2, h2 = low_mid_high(100*attack_rate.values)
	axes[1].fill_between(plot_time,l0,h0,color="#4D648D",alpha=0.1)
	axes[1].fill_between(plot_time,l1,h1,color="#4D648D",alpha=0.3)
	axes[1].fill_between(plot_time,l2,h2,color="#4D648D",alpha=0.8,label="Estimated cumulative incidence using the transmission model")
	axes[1].plot(plot_time,(100*attack_rate.values).mean(axis=0),color="#4D648D",lw=3)
	axes[1].legend(loc=2,frameon=False,fontsize=28)
	axes[1].set_ylabel(r"COVID-19 cumulative incidence (%)")
	axes[1].set_xlim(("2020-02-01",None))

	## Finalize the main figure
	fig.tight_layout()

	## Add a reporting inset
	inset_dim = [0.13, 0.21, 0.24, 0.19] ## left, bottom, width, height
	axes2 = fig.add_axes(inset_dim)
	axes2.spines["left"].set_visible(False)
	axes2.spines["top"].set_visible(False)
	axes2.spines["right"].set_visible(False)

	## Compute a KDE for the cumulative reporting rate
	kde = gaussian_kde(cum_rr_samples.values)
	rr = np.linspace(0,50,1000)
	hist = kde(rr)
	axes2.fill_between(rr,0,hist,color="grey",alpha=0.4)
	axes2.plot(rr,hist,lw=4,color="k")
	axes2.set_ylim((0,None))
	axes2.set_yticks([])
	axes2.set_xlabel("Percent of incidence reported\nto the WDRS")

	## Save the output
	fig.savefig("../_plots/output.png")

	## Done
	plt.show()
