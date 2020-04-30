## File to tell Python this is a library and to set the
## default plotting environment for analyses in this project.

import matplotlib.pyplot as plt

## Custom global matplotlib parameters
## see http://matplotlib.org/users/customizing.html for details.
plt.rcParams["font.size"] = 24.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Garamond","Time New Roman"]
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"

## Get matplotlib ready for the pandas datetimes
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()