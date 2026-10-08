#!/bin/sh

# Runs in the currently active conda env
echo "Using current conda env ($CONDA_DEFAULT_ENV)"

python leafstats_example_3channels.py && echo "3channels done" &
python leafstats_example_1channel.py && echo "1channel done" &
python leafstats_syntheticdata.py && echo "synthetic done" &

wait

echo "all done"

