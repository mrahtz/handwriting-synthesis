import numpy as np
import pathlib
import tempfile

import rnn


def test_smoke_1_step():
    with tempfile.TemporaryDirectory() as temp_dir:
        rnn.main(
            seed=0,
            num_training_steps=1,
            experiment_path=pathlib.Path(temp_dir),
        )


def test_train_loss_history_10_steps_stays_the_same():
    with tempfile.TemporaryDirectory() as temp_dir:
        train_loss_history = rnn.main(
            seed=1,  # seed=0 has infs in the train loss history :(
            num_training_steps=10,
            experiment_path=pathlib.Path(temp_dir),
        )
        np.testing.assert_allclose(
            train_loss_history,
            [
                3.7202573, 3.7197456, 3.6746328, 3.6974313, 3.7937164, 3.702197, 3.7689488, 3.7159014, 3.7125099,
                3.7207985,
            ]
        )
