#!/bin/bash

rm ~/Documents/IFNmodeling/ifnscripts/ifnclass
rm ~/Documents/IFNmodeling/ifnscripts/ifnmodels
rm ~/Documents/IFNmodeling/ifnscripts/ifndatabase

ln -s ~/Documents/IFNmodeling/ifndatabase ~/Documents/IFNmodeling/ifnscripts/ifndatabase 
ln -s ~/Documents/IFNmodeling/ifnmodels ~/Documents/IFNmodeling/ifnscripts/ifnmodels
ln -s ~/Documents/IFNmodeling/ifnclass ~/Documents/IFNmodeling/ifnscripts/ifnclass

rm ~/Documents/IFNmodeling/pydream/GAB/ifndatabase
rm ~/Documents/IFNmodeling/pydream/GAB/ifnmodels
rm ~/Documents/IFNmodeling/pydream/GAB/ifnclass

ln -s ~/Documents/IFNmodeling/ifndatabase ~/Documents/IFNmodeling/pydream/GAB/ifndatabase 
ln -s ~/Documents/IFNmodeling/ifnmodels ~/Documents/IFNmodeling/pydream/GAB/ifnmodels
ln -s ~/Documents/IFNmodeling/ifnclass ~/Documents/IFNmodeling/pydream/GAB/ifnclass

rm ~/Documents/IFNmodeling/ifnscripts/NIH/ifndatabase 
rm ~/Documents/IFNmodeling/ifnscripts/NIH/ifnmodels
rm ~/Documents/IFNmodeling/ifnscripts/NIH/ifnclass

ln -s ~/Documents/IFNmodeling/ifndatabase ~/Documents/IFNmodeling/ifnscripts/NIH/ifndatabase 
ln -s ~/Documents/IFNmodeling/ifnmodels ~/Documents/IFNmodeling/ifnscripts/NIH/ifnmodels
ln -s ~/Documents/IFNmodeling/ifnclass ~/Documents/IFNmodeling/ifnscripts/NIH/ifnclass

rm ~/Documents/IFNmodeling/ifnscripts/EpoR/ifndatabase 
rm ~/Documents/IFNmodeling/ifnscripts/EpoR/ifnmodels
rm ~/Documents/IFNmodeling/ifnscripts/EpoR/ifnclass

ln -s ~/Documents/IFNmodeling/ifndatabase ~/Documents/IFNmodeling/ifnscripts/EpoR/ifndatabase 
ln -s ~/Documents/IFNmodeling/ifnmodels ~/Documents/IFNmodeling/ifnscripts/EpoR/ifnmodels
ln -s ~/Documents/IFNmodeling/ifnclass ~/Documents/IFNmodeling/ifnscripts/EpoR/ifnclass