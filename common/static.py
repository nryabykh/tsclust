start_caption = """When dealing with time series, it would be very useful to identify similar time series and group them in 
some way. One might imagine a lot of purposes for that. 

I used time series clustering for anomaly detection and predictive maintenance of complex objects in my practice. For 
example, if each time series represents the data from some sensor of a complex object, and the number of these 
sensors is quite large, then you can define the normal functionality of the object with a constant set of such 
clusters. If you see, that this clustering structure varies in some way, then it might be a predictor of some 
failures or other kinds of the improper operation mode of the object. 

Another purpose of time series clustering is dimensionality reduction while fitting ML models. If you use time series 
as model features, in some cases it would be better to reduce their number by combining similar series to the new 
one. It will likely accelerate model fitting and prediction and helps to avoid overfitting. """

upload_caption = """Please select the CSV file with time series. The application will try to do the following 
automatically: 

- identify a column with a timestamp;
- choose wide or long table format (both formats supported);
- select columns with time series (if wide format);
- select a column with values and a column with a series name (if long format).

The results of automatic extraction will be shown as default values in the widgets below. Please, change them to the 
correct ones, if needed. """

dbscan_eps_help = """DBSCAN algorithm is used for clustering. You can change the '_epsilon_' parameter of DBSCAN with 
this slider. You could consider '_epsilon_' as maximum allowed distance between the objects of the same cluster. 
Mutual Pearson correlations subtracted from 1 are taken as distances between time series. More details about DBSCAN 
you can find there: https://scikit-learn.org/stable/modules/clustering.html#dbscan """

dbscan_min_samples_help = """DBSCAN algorithm is used for clustering. You can change the '_min_samples_' parameter of 
DBSCAN with this slider. You could consider _min_samples_ as a minimum required number of samples in cluster. If 
some samples are close together (mutual distances are less than _epsilon_), but their number is less than 
_min_samples_, these points won't be clustered. More details about DBSCAN you can find there: 
https://scikit-learn.org/stable/modules/clustering.html#dbscan """

projection_caption = """Each point on the plane represents the single time series. Coordinates of points are calculated 
using the MDS (multidimensional scaling) algorithm. This algorithms projects objects to the plane keeping 
specified distances between objects. Mutual Pearson correlations subtracted from 1 are taken as distances between 
time series. This scatter plot helps estimating appropriate number of clusters."""

default_eps = 0.3
default_min_samples = 3

date_col = 'dt'
