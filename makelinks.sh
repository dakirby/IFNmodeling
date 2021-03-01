#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

rm "$DIR/ifnscripts/ifnclass"
rm $DIR/ifnscripts/ifnmodels
rm $DIR/ifnscripts/ifndatabase

ln -s $DIR/ifndatabase $DIR/ifnscripts/ifndatabase
ln -s $DIR/ifnmodels $DIR/ifnscripts/ifnmodels
ln -s $DIR/ifnclass $DIR/ifnscripts/ifnclass

rm $DIR/pydream/GAB/ifndatabase
rm $DIR/pydream/GAB/ifnmodels
rm $DIR/pydream/GAB/ifnclass

ln -s $DIR/ifndatabase $DIR/pydream/GAB/ifndatabase
ln -s $DIR/ifnmodels $DIR/pydream/GAB/ifnmodels
ln -s $DIR/ifnclass $DIR/pydream/GAB/ifnclass

rm $DIR/ifnscripts/NIH/ifndatabase
rm $DIR/ifnscripts/NIH/ifnmodels
rm $DIR/ifnscripts/NIH/ifnclass

ln -s $DIR/ifndatabase $DIR/ifnscripts/NIH/ifndatabase
ln -s $DIR/ifnmodels $DIR/ifnscripts/NIH/ifnmodels
ln -s $DIR/ifnclass $DIR/ifnscripts/NIH/ifnclass

rm $DIR/ifnscripts/EpoR/ifndatabase
rm $DIR/ifnscripts/EpoR/ifnmodels
rm $DIR/ifnscripts/EpoR/ifnclass

ln -s $DIR/ifndatabase $DIR/ifnscripts/EpoR/ifndatabase
ln -s $DIR/ifnmodels $DIR/ifnscripts/EpoR/ifnmodels
ln -s $DIR/ifnclass $DIR/ifnscripts/EpoR/ifnclass
