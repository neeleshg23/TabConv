import amm 
import vquantizers as vq
from vq_amm import VQMatmul

class PQ_AMM_CNN(VQMatmul):
    def _create_encoder(self, ncodebooks):  # to be overriden by subclasses
        return vq.PQEncoder_CNN(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            **self._get_encoder_kwargs())

    def _get_ncentroids(self):
        return 256

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else A.shape[0] * A.shape[1] * self.ncentroids
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        nlookups = A.shape[0] * B.shape[1] * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, amm.KEY_NLOOKUPS: nlookups}

