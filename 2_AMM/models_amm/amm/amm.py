import abc
import numpy as np

class InvalidParametersException(Exception):
    pass


class ApproxMatmul(abc.ABC):

    def __init__(*args_unused, **kwargs_unused):
        pass

    def fit(self, A, B, Y=None):  # Y = A @ B if not specified
        pass

    def set_A(self, A):
        pass

    def set_B(self, B):
        pass

    def reset_for_new_task(self):
        pass

    @abc.abstractmethod
    def __call__(self, A, B):
        pass

    def predict(self, A, B):
        return self(A, B)

    def get_params(self):
        return {}

    # def get_nmuls(self, A, B, fixedA=False, fixedB=False):
    @abc.abstractmethod
    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        pass
