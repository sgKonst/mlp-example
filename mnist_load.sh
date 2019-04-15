#!/bin/bash

BASE_URL="http://yann.lecun.com/exdb/mnist"
NAMES="train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz"

mkdir mnist_data

for f in $NAMES; do
    echo "Loading $f"
    curl "$BASE_URL/$f" --output "$(pwd)/mnist_data/$f"
done

cd mnist_data
gzip *ubyte.gz -d
