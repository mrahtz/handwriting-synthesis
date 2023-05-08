#!/bin/bash

docker run \
    -it \
    --gpus=all \
    -v $(pwd):/workspace \
    -w /workspace \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    sjvasquez-handwriting-synthesis \
    "$@"
