'''
kingcount_rainier installation. Requirements are listed in requirements.txt. Quick install by:
    python setup.py develop          # standard install
'''

from setuptools import setup, find_packages

setup(name='kingcounty_rainier',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          # public dependencies
          "matplotlib",
          "numpy",
          "scipy",
          "tqdm",
          "pandas",
          "geopandas",
          # pypi-production dependencies
          "gdal==2.4.1",
          "fiona==1.8.6",
      ],
      dependency_links=["https://packages.idmod.org/api/pypi/pypi-production/simple/GDAL",
                        "https://packages.idmod.org/api/pypi/pypi-production/simple/Fiona"],
      )
