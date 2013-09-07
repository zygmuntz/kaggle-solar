kaggle-solar
============

A repo for [Solar Energy Prediction Contest](http://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest) at Kaggle.

No article at [http://fastml.com/](http://fastml.com/) yet.

	starter_code_mod.py - modified starter code from Alec Radford.
	
The main difference from the original is twofold:

* we do not average over hours
* we use fewer GEFS points for predictions

The first factor increases dimensionality five times, the second reduces dimensionality 3.6 times (10x4 grid instead of 16x9), so overall dimensionality goes up by 38%, to 3000. We deal with by increasing regularization coefficient. Validation error goes from 226k to 223k. A public leaderboard score is 221.5k.

