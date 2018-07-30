from setuptools import setup, find_packages

setup(name="PyCausality",
      version="0.0.3",
      python_requires=">3.5.2",
      description="Python package for detection and quantification \
                    of statistical causality between time series",
      author="Zac Keskin",
      author_email="Zac.Keskin.17@ucl.ac.uk",
      maintainer="Zac Keskin",
      maintainer_email="zackeskin@outlook.com",
      url="https://github.com/ZacKeskin/PyCausality",
      license="GNU GPLv3",
      packages=find_packages(),
      install_requires = ['pandas','statsmodels','numpy', 'python-dateutil','nose']
)