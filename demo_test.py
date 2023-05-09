import pathlib
import tempfile

import numpy as np

import demo


def test_sample():
    with tempfile.TemporaryDirectory() as temp_dir:
        hand = demo.Hand(
            log_dir=temp_dir,
            prediction_dir=temp_dir,
            checkpoint_dir='checkpoints/my_run',
        )
        samples = hand._sample(lines=['foo'])
        np.testing.assert_allclose(
            samples,
            np.array([[
                [-2.1618712e+00, -1.6633049e+00, 0.0000000e+00],
                [-1.3871622e+00, -1.8977870e+00, 0.0000000e+00],
                [-5.7453996e-01, -1.9552549e+00, 0.0000000e+00],
                [5.5454880e-01, -1.8334516e+00, 0.0000000e+00],
                [1.2930764e+00, -1.5802771e+00, 0.0000000e+00],
                [1.4939109e+00, -1.3240921e+00, 0.0000000e+00],
                [1.3807520e+00, -1.0566163e+00, 0.0000000e+00],
                [1.1644899e+00, -7.8023142e-01, 0.0000000e+00],
                [8.1853122e-01, -3.9360392e-01, 0.0000000e+00],
                [4.4519031e-01, -1.2225797e-01, 0.0000000e+00],
                [1.4789569e-01, 7.1995459e-02, 1.0000000e+00],
                [-1.0670574e+01, 3.0524211e+00, 1.0000000e+00],
                [4.9286351e+00, 3.0346632e-01, 0.0000000e+00],
                [1.3143038e+00, 1.8690088e-01, 0.0000000e+00],
                [1.7202193e+00, 3.8036561e-01, 1.0000000e+00],
                [4.6639252e+00, -1.4219725e-01, 0.0000000e+00],
                [2.4817877e-02, -1.1743829e-01, 0.0000000e+00],
                [-3.7754912e-02, -7.4919738e-02, 0.0000000e+00],
                [-1.1867904e-01, -7.0123971e-03, 0.0000000e+00],
                [-2.3380674e-01, 2.2964138e-01, 0.0000000e+00],
                [-3.8184190e-01, 4.4711801e-01, 0.0000000e+00],
                [-5.5737245e-01, 5.4319137e-01, 0.0000000e+00],
                [-7.3006636e-01, 4.2782146e-01, 0.0000000e+00],
                [-7.9094625e-01, 1.4036030e-01, 0.0000000e+00],
                [-6.0132033e-01, -3.1895828e-01, 0.0000000e+00],
                [-3.3075914e-01, -4.7342139e-01, 0.0000000e+00],
                [-8.7924547e-02, -4.1750759e-01, 1.0000000e+00],
                [7.5901470e+00, -5.0309867e-01, 0.0000000e+00],
                [-1.8060082e-01, 1.8885353e-01, 0.0000000e+00],
                [-3.1834459e-01, 3.1112391e-01, 0.0000000e+00],
                [-4.2278379e-01, 3.6325043e-01, 0.0000000e+00],
                [-4.7120383e-01, 2.1398149e-01, 0.0000000e+00],
                [-5.0069362e-01, -1.0402697e-01, 0.0000000e+00],
                [-4.5800763e-01, -4.6660209e-01, 0.0000000e+00],
                [-3.3018348e-01, -7.9804987e-01, 0.0000000e+00],
                [-8.7471738e-02, -9.9889350e-01, 0.0000000e+00],
                [2.8853402e-01, -9.6033776e-01, 0.0000000e+00],
                [6.5827852e-01, -7.3382860e-01, 0.0000000e+00],
                [9.5913696e-01, -3.3861947e-01, 0.0000000e+00],
                [1.0978674e+00, 1.1025161e-01, 0.0000000e+00],
                [1.0109771e+00, 5.1884139e-01, 0.0000000e+00],
                [7.0293903e-01, 8.2164693e-01, 0.0000000e+00],
                [2.5889444e-01, 9.9686998e-01, 0.0000000e+00],
                [-2.4003291e-01, 9.7305459e-01, 0.0000000e+00],
                [-6.7661721e-01, 8.0571276e-01, 0.0000000e+00],
                [-9.8492736e-01, 5.0857705e-01, 0.0000000e+00],
                [-9.1273183e-01, 2.0884109e-01, 0.0000000e+00],
            ]]),
        )