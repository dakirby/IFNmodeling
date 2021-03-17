#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python $DIR/pydream/GAB/run_PyDREAM.py

python $DIR/figure_2.py
python $DIR/figure_3.py
python $DIR/figure_4.py
python $DIR/figure_5.py
