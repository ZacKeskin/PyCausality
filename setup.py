from setuptools import setup, find_packages

setup(name="PyCausality",
      version="1.2.0",
      python_requires=">3.5.2",
      description=" Extended significance testing to linear TE calculations",
      long_description="Python package for detection and quantification \
                    of statistical causality between time series, using information \
                    theoretic models. See https://github.com/ZacKeskin/PyCausality for details",
      author="Zac Keskin",
      author_email="Zac.Keskin.17@ucl.ac.uk",
      maintainer="Zac Keskin",
      maintainer_email="zackeskin@outlook.com",
      url="https://github.com/ZacKeskin/PyCausality",
      license="GNU GPLv3",
      packages=find_packages(),
      install_requires = ['pandas','statsmodels','numpy', 'python-dateutil==2.6.1','nose'] 
      ## Technically also needs Matplotlib for the plot_pdf() function, however pip chooses
      #  to install its own pip which is a nightmare on OSX. Safer to let the users
      # manage their own matplotlib installation, since this is a pretty standard package and
      # plot_pdf() is in any case only an auxiliary function
     )