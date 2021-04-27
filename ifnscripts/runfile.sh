#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

while getopts ":f" opt; do
  case $opt in
  f)
      cd $DIR/..
      cd pydream/GAB
      python3 run_PyDREAM.py
      cd $DIR
      echo "DONE FITTING"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

python3 $DIR/DR_figure.py                # Figure 2
python3 $DIR/cell_size_figure.py         # Figure 3
python3 $DIR/negative_feedback_figure.py # Figure 3
python3 $DIR/K4_and_TC_figures.py        # Figure 4 & 5
python3 $DIR/AP_AV_figure.py             # Figure 6
