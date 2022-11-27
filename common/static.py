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
