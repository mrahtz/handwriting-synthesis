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
        samples = hand._sample(lines=['foo', 'bar baz'])
        np.testing.assert_allclose(
            samples[0],
            np.array([
                [-2.162, -1.664, 0.],
                [-0.712, -1.564, 0.],
                [-0.346, -2.002, 0.],
                [-0.053, -2.554, 0.],
                [0.169, -2.766, 0.],
                [0.185, -2.673, 0.],
                [0.294, -2.332, 0.],
                [0.364, -1.894, 0.],
                [0.353, -1.447, 0.],
                [0.225, -0.896, 0.],
                [0.098, -0.397, 1.],
                [-3.436, 8.901, 0.],
                [0.326, 0.079, 0.],
                [0.785, 0.106, 0.],
                [0.937, 0.091, 0.],
                [0.78, 0.045, 1.],
                [8.15, 2.782, 0.],
                [-0.452, -0.001, 0.],
                [-0.724, -0.188, 0.],
                [-0.886, -0.444, 0.],
                [-0.829, -0.749, 0.],
                [-0.682, -0.935, 0.],
                [-0.429, -1.032, 0.],
                [-0.118, -0.981, 0.],
                [0.186, -0.89, 0.],
                [0.464, -0.632, 0.],
                [0.702, -0.283, 0.],
                [0.885, 0.133, 0.],
                [0.983, 0.58, 0.],
                [0.948, 0.961, 0.],
                [0.775, 1.112, 0.],
                [0.504, 1.146, 0.],
                [0.171, 1., 0.],
                [-0.198, 0.758, 0.],
                [-0.537, 0.396, 0.],
                [-0.608, 0.093, 0.],
                [-0.48, -0.098, 1.],
                [6.757, 0.177, 0.],
                [-0.268, 0.094, 0.],
                [-0.449, -0.054, 0.],
                [-0.595, -0.325, 0.],
                [-0.621, -0.69, 0.],
                [-0.603, -0.959, 0.],
                [-0.433, -1.056, 0.],
                [-0.162, -0.999, 0.],
                [0.177, -0.817, 0.],
                [0.54, -0.51, 0.],
                [0.858, -0.118, 0.],
                [1.053, 0.287, 0.],
                [1.118, 0.618, 0.],
                [0.986, 0.896, 0.],
                [0.731, 1.031, 0.],
                [0.387, 1.069, 0.],
                [0.025, 1.012, 0.],
                [-0.323, 0.812, 0.],
                [-0.632, 0.505, 0.],
                [-0.673, 0.244, 0.],
                [-0.504, 0.037, 1.],
            ]),
            atol = 1e-3,
        )
        np.testing.assert_allclose(
            samples[1],
            np.array([
                [-0.045, -0.053, 0.],
                [-0.044, -0.062, 0.],
                [-0.036, -0.08, 0.],
                [0.004, -0.099, 0.],
                [0.015, -0.314, 0.],
                [-0.011, -0.61, 0.],
                [-0.026, -0.966, 0.],
                [-0.084, -1.241, 0.],
                [-0.138, -1.403, 0.],
                [-0.193, -1.467, 0.],
                [-0.234, -1.531, 0.],
                [-0.256, -1.508, 0.],
                [-0.223, -1.322, 0.],
                [-0.15, -1.039, 0.],
                [-0.041, -0.652, 0.],
                [0.011, -0.326, 0.],
                [0.019, -0.055, 0.],
                [0.031, 0.285, 0.],
                [0.057, 0.609, 0.],
                [0.161, 0.938, 0.],
                [0.366, 1.206, 0.],
                [0.594, 1.322, 0.],
                [0.85, 1.255, 0.],
                [1.071, 1.096, 0.],
                [1.251, 0.821, 0.],
                [1.347, 0.488, 0.],
                [1.35, 0.094, 0.],
                [1.208, -0.296, 0.],
                [0.924, -0.712, 0.],
                [0.536, -1.039, 0.],
                [0.095, -1.228, 0.],
                [-0.34, -1.314, 0.],
                [-0.763, -1.221, 0.],
                [-1.106, -1.037, 0.],
                [-1.364, -0.746, 0.],
                [-1.426, -0.425, 0.],
                [-1.342, -0.101, 0.],
                [-1.084, 0.289, 0.],
                [-0.716, 0.455, 0.],
                [-0.354, 0.399, 1.],
                [15.223, 6.202, 0.],
                [-0.012, 0.206, 0.],
                [-0.068, 0.305, 0.],
                [-0.198, 0.357, 0.],
                [-0.374, 0.252, 0.],
                [-0.548, 0.085, 0.],
                [-0.711, -0.135, 0.],
                [-0.84, -0.389, 0.],
                [-0.88, -0.668, 0.],
                [-0.851, -0.896, 0.],
                [-0.696, -1.08, 0.],
                [-0.456, -1.116, 0.],
                [-0.191, -1.057, 0.],
                [0.09, -0.884, 0.],
                [0.333, -0.629, 0.],
                [0.514, -0.313, 0.],
                [0.667, 0.047, 0.],
                [0.73, 0.437, 0.],
                [0.761, 0.766, 0.],
                [0.756, 1.015, 0.],
                [0.692, 1.128, 0.],
                [0.553, 1.124, 0.],
                [0.391, 0.955, 0.],
                [0.208, 0.713, 0.],
                [0.088, 0.433, 0.],
                [-0., 0.101, 0.],
                [-0.036, -0.294, 0.],
                [-0.072, -0.661, 0.],
                [-0.083, -1.016, 0.],
                [-0.055, -1.276, 0.],
                [-0.002, -1.367, 0.],
                [0.088, -1.28, 0.],
                [0.219, -1.048, 0.],
                [0.36, -0.668, 0.],
                [0.541, -0.224, 0.],
                [0.724, 0.228, 0.],
                [0.837, 0.666, 0.],
                [0.882, 1.003, 0.],
                [0.821, 1.218, 0.],
                [0.681, 1.235, 0.],
                [0.517, 1.056, 0.],
                [0.347, 0.754, 0.],
                [0.204, 0.401, 0.],
                [0.089, -0.009, 0.],
                [-0.041, -0.407, 0.],
                [-0.151, -0.819, 0.],
                [-0.284, -1.141, 0.],
                [-0.392, -1.289, 0.],
                [-0.42, -1.245, 0.],
                [-0.357, -0.958, 0.],
                [-0.259, -0.575, 0.],
                [-0.127, -0.124, 0.],
                [0.011, 0.344, 0.],
                [0.129, 0.808, 0.],
                [0.271, 1.132, 0.],
                [0.429, 1.39, 0.],
                [0.586, 1.414, 0.],
                [0.731, 1.315, 0.],
                [0.852, 1.09, 0.],
                [0.905, 0.773, 0.],
                [0.903, 0.433, 0.],
                [0.84, 0.13, 0.],
                [0.799, -0.183, 0.],
                [0.628, -0.296, 0.],
                [0.395, -0.262, 1.],
                [12.951, 5.292, 0.],
                [-0.067, -0.443, 0.],
                [-0.091, -0.816, 0.],
                [-0.136, -1.166, 0.],
                [-0.188, -1.378, 0.],
                [-0.318, -1.482, 0.],
                [-0.456, -1.551, 0.],
                [-0.558, -1.538, 0.],
                [-0.565, -1.425, 0.],
                [-0.516, -1.123, 0.],
                [-0.358, -0.748, 0.],
                [-0.2, -0.334, 0.],
                [-0.034, 0.052, 0.],
                [0.119, 0.449, 0.],
                [0.298, 0.757, 0.],
                [0.467, 1.027, 0.],
                [0.656, 1.159, 0.],
                [0.791, 1.137, 0.],
                [0.905, 0.987, 0.],
                [0.976, 0.801, 0.],
                [0.998, 0.533, 0.],
                [0.962, 0.206, 0.],
                [0.875, -0.145, 0.],
                [0.697, -0.469, 0.],
                [0.47, -0.76, 0.],
                [0.183, -0.945, 0.],
                [-0.151, -1.031, 0.],
                [-0.492, -1.043, 0.],
                [-0.766, -1.007, 0.],
                [-1.003, -0.863, 0.],
                [-1.149, -0.668, 0.],
                [-1.2, -0.417, 0.],
                [-1.114, -0.146, 0.],
                [-0.91, 0.112, 0.],
                [-0.522, 0.392, 0.],
                [-0.186, 0.474, 0.],
                [0.019, 0.4, 1.],
                [14.237, 4.461, 0.],
                [-0.076, 0.271, 0.],
                [-0.212, 0.35, 0.],
                [-0.419, 0.328, 0.],
                [-0.648, 0.133, 0.],
                [-0.86, -0.085, 0.],
                [-1.053, -0.335, 0.],
                [-1.155, -0.603, 0.],
                [-1.128, -0.842, 0.],
                [-0.959, -0.992, 0.],
                [-0.664, -1.06, 0.],
                [-0.326, -0.987, 0.],
                [0.007, -0.814, 0.],
                [0.298, -0.569, 0.],
                [0.552, -0.258, 0.],
                [0.778, 0.095, 0.],
                [0.947, 0.463, 0.],
                [1.041, 0.802, 0.],
                [1.058, 1.069, 0.],
                [0.983, 1.185, 0.],
                [0.827, 1.178, 0.],
                [0.603, 1.021, 0.],
                [0.349, 0.761, 0.],
                [0.141, 0.475, 0.],
                [-0.043, 0.123, 0.],
                [-0.197, -0.284, 0.],
                [-0.301, -0.701, 0.],
                [-0.396, -1.061, 0.],
                [-0.402, -1.331, 0.],
                [-0.324, -1.366, 0.],
                [-0.173, -1.312, 0.],
                [0.035, -1.135, 0.],
                [0.291, -0.855, 0.],
                [0.552, -0.479, 0.],
                [0.923, 0.041, 0.],
                [0.938, 0.348, 0.],
                [0.724, 0.402, 1.],
                [2.063, 7.423, 0.],
                [0.072, 0.198, 0.],
                [0.271, 0.237, 0.],
                [0.563, 0.201, 0.],
                [0.851, 0.057, 0.],
                [1.028, -0.081, 0.],
                [1.073, -0.232, 0.],
                [0.942, -0.339, 0.],
                [0.713, -0.451, 0.],
                [0.404, -0.557, 0.],
                [0.039, -0.686, 0.],
                [-0.348, -0.809, 0.],
                [-0.688, -0.933, 0.],
                [-0.945, -0.996, 0.],
                [-1.025, -1.023, 0.],
                [-0.932, -0.957, 0.],
                [-0.632, -0.844, 0.],
                [-0.211, -0.697, 0.],
                [0.296, -0.521, 0.],
                [0.798, -0.326, 0.],
                [1.202, -0.123, 0.],
                [1.385, 0.121, 0.],
                [1.342, 0.351, 0.],
                [0.989, 0.392, 0.],
                [0.546, 0.307, 1.],
            ]),
            atol=1e-3,
        )
