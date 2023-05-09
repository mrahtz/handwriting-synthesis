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


def test_smoke_20_steps():
    with tempfile.TemporaryDirectory() as temp_dir:
        rnn.main(
            seed=0,
            num_training_steps=20,
            experiment_path=pathlib.Path(temp_dir),
        )
