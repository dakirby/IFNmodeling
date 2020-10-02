#!/bin/bash

rm ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/ifnclass
rm ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/ifnmodels
rm ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/ifndatabase

ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifndatabase ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/ifndatabase 
ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifnmodels ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/ifnmodels
ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifnclass ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/ifnclass

rm ~/Grad_Studies_Year_4/IFNmodeling/pydream/GAB/ifndatabase
rm ~/Grad_Studies_Year_4/IFNmodeling/pydream/GAB/ifnmodels
rm ~/Grad_Studies_Year_4/IFNmodeling/pydream/GAB/ifnclass

ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifndatabase ~/Grad_Studies_Year_4/IFNmodeling/pydream/GAB/ifndatabase 
ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifnmodels ~/Grad_Studies_Year_4/IFNmodeling/pydream/GAB/ifnmodels
ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifnclass ~/Grad_Studies_Year_4/IFNmodeling/pydream/GAB/ifnclass

rm ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/NIH/ifndatabase 
rm ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/NIH/ifnmodels
rm ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/NIH/ifnclass

ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifndatabase ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/NIH/ifndatabase 
ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifnmodels ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/NIH/ifnmodels
ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifnclass ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/NIH/ifnclass

rm ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/EpoR/ifndatabase 
rm ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/EpoR/ifnmodels
rm ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/EpoR/ifnclass

ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifndatabase ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/EpoR/ifndatabase 
ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifnmodels ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/EpoR/ifnmodels
ln -s ~/Grad_Studies_Year_4/IFNmodeling/ifnclass ~/Grad_Studies_Year_4/IFNmodeling/ifnscripts/EpoR/ifnclass
