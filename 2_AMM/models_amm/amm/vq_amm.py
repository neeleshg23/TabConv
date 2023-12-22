
import abc
import numpy as np

import vquantizers as vq
import amm

KEY_NLOOKUPS = 'nlookups'


class VQMatmul(amm.ApproxMatmul, abc.ABC):
    def __init__(self, ncodebooks, ncentroids=None):
        self.ncodebooks = ncodebooks
        self.ncentroids = (self._get_ncentroids() if ncentroids is None
                           else ncentroids)
        self.enc = self._create_encoder(ncodebooks)
        self.reset_for_new_task()

    @abc.abstractmethod
    def _create_encoder(self, ncodebooks):  # to be overriden by subclasses
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            **self._get_encoder_kwargs())

    # @abc.abstractmethod
    def _get_ncentroids(self):
        pass

    @abc.abstractmethod
    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        pass

    def _get_encoder_kwargs(self):  # to be overriden by subclasses
        return {}

    def reset_for_new_task(self):
        self.A_enc = None
        self.luts = None

    def fit(self, A, B, Y=None):
        _, D = A.shape
        if D < self.ncodebooks:
            raise amm.InvalidParametersException(
                'D < k: {} < {}'.format(D, self.ncodebooks))
        self.enc.fit(A, B.T)

    def set_A(self, A):
        self.A_enc = self.enc.encode_X(A)

    def set_B(self, B):
        self.luts = self.enc.encode_Q(B.T)

    def __call__(self, A, B):
        if self.A_enc is None:
            self.set_A(A)
        if self.luts is None:
            self.set_B(B)
        return self.enc.dists_enc(self.A_enc, self.luts)

    def predict_cnn(self, A, B):
        if self.A_enc is None:
            self.A_enc = self.enc.encode_X(A)
        if self.luts is None:
            self.luts = self.enc.encode_Q(B.T)
        return self.enc.dists_enc_cnn(self.A_enc, self.luts)

    def get_params(self):
        return {'ncodebooks': self.ncodebooks}

# ================================================================ PQ

class PQMatmul(VQMatmul):

    def _create_encoder(self, ncodebooks):  # to be overriden by subclasses
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            **self._get_encoder_kwargs())

    def _get_ncentroids(self):
        #mark:return 256
        #works.
        return 16

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else A.shape[0] * A.shape[1] * self.ncentroids
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        nlookups = A.shape[0] * B.shape[1] * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}

