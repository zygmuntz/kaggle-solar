"""
AMS Solar Energy Prediction Starter Code!

Some code to help get people off the ground and point them in a decent direction to go.
Requires scikit-learn, numpy, and netCDF4.
It's where I started and what I've built off of.
Some of this code is recycled for Miroslaw and Paul Duan's forum code from the Amazon Challenge, thanks!

Email's alec.radford@gmail.com if you have questions.

---

Modified by zygmunt@fastml.com

original:
Best alpha of 0.1 with mean average error of 2260498.69318

fewer points (different alphas):
Best alpha of 0.0138949549437 with mean average error of 2263705.32851

w/hours:
Best alpha of 0.193069772888 with mean average error of 2241100.4674

w/hours, finer alpha search:
Best alpha of 0.3 with mean average error of 2237411.22884

"""

import csv
import os
import netCDF4 as nc
import numpy as np 
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split

SEED = 42 # Random seed to keep consistent

'''
Loads a list of GEFS files merging them into model format.
'''
def load_GEFS_data(directory,files_to_use,file_sub_str):
	for i,f in enumerate(files_to_use):
		if i == 0:
			X = load_GEFS_file(directory,files_to_use[i],file_sub_str)
		else:
			X_new = load_GEFS_file(directory,files_to_use[i],file_sub_str)
			X = np.hstack((X,X_new))
	return X

'''
Loads GEFS file using specified merge technique.
'''
def load_GEFS_file(directory,data_type,file_sub_str):
	print 'loading',data_type
	path = os.path.join(directory,data_type+file_sub_str)
	X = nc.Dataset(path,'r+').variables.values()[-1][:,:,:,3:7,3:13] # get rid of some GEFS points
	#X = X.reshape(X.shape[0],55,4,10) 								 # Reshape to merge sub_models and time_forcasts
	X = np.mean(X,axis=1) 											 # Average models, but not hours
	X = X.reshape(X.shape[0],np.prod(X.shape[1:])) 					 # Reshape into (n_examples,n_features)
	return X

'''
Load csv test/train data splitting out times.
'''
def load_csv_data(path):
	data = np.loadtxt(path,delimiter=',',dtype=float,skiprows=1)
	times = data[:,0].astype(int)
	Y = data[:,1:]
	return times,Y

'''
Saves out to a csv.
Just reads in the example csv and writes out 
over the zeros with the model predictions.
'''
def save_submission(preds,submit_name,data_dir):
	fexample = open(os.path.join(data_dir,'sampleSubmission.csv'))
	fout = open(submit_name,'wb')
	fReader = csv.reader(fexample,delimiter=',', skipinitialspace=True)
	fwriter = csv.writer(fout)
	for i,row in enumerate(fReader):
		if i == 0:
			fwriter.writerow(row)
		else:
			row[1:] = preds[i-1]
			fwriter.writerow(row)
	fexample.close()
	fout.close()

'''
Get the average mean absolute error for models trained on cv splits
'''
def cv_loop(X, y, model, N):
    MAEs = 0
    for i in range(N):
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=.20, random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict(X_cv)
        mae = metrics.mean_absolute_error(y_cv,preds)
        print "MAE (fold %d/%d): %f" % (i + 1, N, mae)
        MAEs += mae
    return MAEs/N

'''
Everything together - print statements describe what's happening
'''
def main(data_dir='./data/',N=10,cv_test_size=0.2,files_to_use='all',submit_name='submission.csv'):
	if files_to_use == 'all':
		files_to_use = ['dswrf_sfc','dlwrf_sfc','uswrf_sfc','ulwrf_sfc','ulwrf_tatm','pwat_eatm','tcdc_eatm','apcp_sfc','pres_msl','spfh_2m','tcolc_eatm','tmax_2m','tmin_2m','tmp_2m','tmp_sfc']
	train_sub_str = '_latlon_subset_19940101_20071231.nc'
	test_sub_str = '_latlon_subset_20080101_20121130.nc'

	print 'Loading training data...'
	trainX = load_GEFS_data(data_dir,files_to_use,train_sub_str)
	times,trainY = load_csv_data(os.path.join(data_dir,'train.csv'))
	print 'Training data shape',trainX.shape,trainY.shape

	# Gotta pick a scikit-learn model
	model = Ridge(normalize=True) # Normalizing is usually a good idea

	print 'Finding best regularization value for alpha...'
	alphas = np.logspace(-3,1,8,base=10) # List of alphas to check
	alphas = np.array(( 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ))
	maes = []
	for alpha in alphas:
		model.alpha = alpha
		mae = cv_loop(trainX,trainY,model,N)
		maes.append(mae)
		print 'alpha %.4f mae %.4f' % (alpha,mae)
	best_alpha = alphas[np.argmin(maes)]
	print 'Best alpha of %s with mean average error of %s' % (best_alpha,np.min(maes))

	print 'Fitting model with best alpha...'
	model.alpha = best_alpha
	model.fit(trainX,trainY)

	print 'Loading test data...'
	testX = load_GEFS_data(data_dir,files_to_use,test_sub_str)
	print 'Test data shape',testX.shape

	print 'Predicting...'
	preds = model.predict(testX)

	print 'Saving to csv...'
	save_submission(preds,submit_name,data_dir)

if __name__ == "__main__":
	args = { 'data_dir':  './train/', # Set to your data directory assumes all data is in there - no nesting
		'N': 5,                      # Amount of CV folds
		'cv_test_size': 0.2,         # Test split size in cv
		'files_to_use': 'all',       # Choices for files_to_use: the string all, or a list of strings corresponding to the unique part of a GEFS filename
		'submit_name': 'submission_mod_whours.csv'
	}
	main(**args)

"""
Big scary license:

The MIT License (MIT)

Copyright (c) 2013 Alec Radford

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

tl;dr version:

You can do whatever you want just don't sue me! =P
"""