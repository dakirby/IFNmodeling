#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

while getopts ":a" opt; do
  case $opt in
  fit)
      python $DIR/pydream/GAB/run_PyDREAM.py
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

python $DIR/figure_2.py
python $DIR/figure_3.py
python $DIR/figure_4.py
python $DIR/figure_5.py
