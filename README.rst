=======
RAINIER
=======

RAINIER is a statistical approach for fitting SEIR epidemic models to case and mortality
data. This example demonstrates the approach on COVID-19 data from King County, WA.

For detailed information, see the report on King County associated with this code. [LINK]

.. contents:: Contents
   :local:
   :depth: 2


Requirements
============

We did the analysis using Python 3.6.10. The following Python packages are also required:

*  numpy (1.14.5)
*  scipy (1.1.0)
*  pandas (0.25.0)
*  matplotlib (3.1.0)
*  tqdm (4.26.0)

Recommended Install Instructions
================================

1. Clone a copy of the repository.

2. (Optional) Create and activate a virtual environment.

3. Navigate to the root of the repository and install with:
        ::

          python setup.py develop

This will install all the required Python package dependencies.

Running RAINIER
---------------

1. From the report_april26 directory, run
2. > python <Report_Script_Name>.py (e.g. RAINIER.py)


Licenses
========

See the LICENSE file for more information.


Disclaimer
==========

The code in this repository was developed by IDM to support our research in
disease transmission and managing epidemics. We’ve made it publicly available
under the Creative Commons Attribution-Noncommercial-ShareAlike 4.0 License to
provide others with a better understanding of our research and an opportunity to
build upon it for their own work. We make no representations that the code works
as intended or that we will provide support, address issues that are found, or
accept pull requests. You are welcome to create your own fork and modify the
code to suit your own modeling needs as contemplated under the Creative Commons
Attribution-Noncommercial-ShareAlike 4.0 License.
