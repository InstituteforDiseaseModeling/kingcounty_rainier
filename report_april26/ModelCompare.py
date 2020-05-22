""" ModelCompare.py

Compare the previous KC reports' statistical model output to the new version 
based on updated data. """
import sys
from pathlib import Path
sys.path.append(Path("../"))
import warnings

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(6)

## For plot settings
import rainier

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

if __name__ == "__main__":

	## Get the second report
	kc2_points = pd.read_pickle("../_outputs/r0_4_10.pkl")
	kc2_m = pd.read_pickle("../_outputs/mobility_r0_4_10.pkl")

	## Get the third report
	kc3_points = pd.read_pickle("../_outputs/r0_4_19.pkl")

	## Get the latest points
	kc4_points = pd.read_pickle("../_outputs/r0_4_26.pkl")

	## Set up a figure
	fig, axes = plt.subplots(figsize=(18,9),sharex=True)
	axes_setup(axes)

	## Plot the mobility model
	fit = kc2_m.loc[:"2020-03-25"]
	forecast = kc2_m.loc["2020-03-26":]
	axes.plot(forecast.index,forecast["low"],"#FAAF08",lw=3,ls="dashed")
	axes.plot(forecast.index,forecast["high"],"#FAAF08",lw=3,ls="dashed",label="Mobility-based nowcast from the 4/10 report")
	
	## Plot the data
	shift = pd.to_timedelta(0.1,unit="d")
	axes.errorbar(kc2_points.index-shift,kc2_points["r0_t"].values,yerr=2.*kc2_points["std_err"],
				  color="0.8",ls="None",marker="o",label=r"R$_e$ estimates based on WDRS compiled on 4/3 (4/10 report)")
	axes.errorbar(kc3_points.index,kc3_points["r0_t"].values,yerr=2.*kc3_points["std_err"],
				  color="0.5",ls="None",marker="o",label=r"R$_e$ estimates based on WDRS compiled on 4/19 (4/21 report)")
	axes.errorbar(kc4_points.index+shift,kc4_points["r0_t"].values,yerr=2.*kc4_points["std_err"],
				  color="k",ls="None",marker="o",label=r"R$_e$ estimates based on WDRS compiled on 4/26 (this report)")

	## Re = 1 line?
	axes.set_ylim((0,5))
	axis_to_data = axes.transAxes + axes.transData.inverted()
	axes.axhline(1.0,c="#4D648D",lw=2,ls="dashed",label=r"Threshold for declining transmission, R$_{e}=1$")
	
	## Details
	axes.legend(loc=1,frameon=False,fontsize=28)
	axes.set_ylabel(r"Effective reproductive number (R$_{e}$)")
	fig.tight_layout()
	fig.savefig(Path("../_plots/model_compare.png"))
	plt.show()
