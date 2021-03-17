#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

while getopts ":f" opt; do
  case $opt in
  f)
      cd $DIR/..
      cd pydream/GAB
      python3 run_PyDREAM.py
      cd $DIR
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
