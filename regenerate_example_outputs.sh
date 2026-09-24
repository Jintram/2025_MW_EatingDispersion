#!/bin/sh

conda activate 2026_leafdamage2

conda run -n 2026_leafdamage2 python leafstats_example_3channels.py && echo "3channels done" &
conda run -n 2026_leafdamage2 python leafstats_example_1channel.py && echo "1channel done" &
conda run -n 2026_leafdamage2 python leafstats_syntheticdata.py && echo "synthetic done" &
wait
echo "all done"

