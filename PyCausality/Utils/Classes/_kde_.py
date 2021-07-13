import numpy as np
from numpy import atleast_2d, pi, sqrt
from scipy import linalg, stats


class _kde_(stats.gaussian_kde):
    """Subclass of scipy.stats.gaussian_kde.

    This is to enable the passage of a pre-defined covariance matrix, via the
    `covar` parameter. This is handled internally within TransferEntropy class.
    The matrix is calculated on the overall dataset, before windowing, which allows for consistency between windows,
    and avoiding duplicative computational operations, compared with calculating the covariance each window.

    Functions left as much as possible identical to scipi.stats.gaussian_kde; docs available:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    """

    def __init__(self, dataset, bw_method=None, df=None, covar=None):
        self.dataset = atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape
        self.set_bandwidth(bw_method=bw_method, covar=covar)

    def set_bandwidth(self, bw_method=None, covar=None):

        if bw_method is None:
            pass
        elif bw_method == "scott":
            self.covariance_factor = self.scotts_factor
        elif bw_method == "silverman":
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, str):
            self._bw_method = "use constant"
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = (
                "`bw_method` should be 'scott', 'silverman', a scalar " "or a callable."
            )
            raise ValueError(msg)

        self._compute_covariance(covar)

    def _compute_covariance(self, covar):

        if covar is not None:
            self._data_covariance = covar
            self._data_inv_cov = linalg.inv(self._data_covariance)

        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, "_data_inv_cov"):
            self._data_covariance = atleast_2d(
                np.cov(self.dataset, rowvar=1, bias=False)
            )
            self._data_inv_cov = linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor ** 2
        self.inv_cov = self._data_inv_cov / self.factor ** 2
        self._norm_factor = sqrt(linalg.det(2 * pi * self.covariance)) * self.n
