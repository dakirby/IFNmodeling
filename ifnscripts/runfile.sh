#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

while getopts ":a" opt; do
  case $opt in
  fit)
      python3 $DIR/pydream/GAB/run_PyDREAM.py
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

python3 $DIR/figure_2.py
python3 $DIR/figure_3.py
python3 $DIR/figure_4.py
python3 $DIR/figure_5.py
