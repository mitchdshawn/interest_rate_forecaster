# US FOMC Communication Interest Rate Forecaster

#### Shawn Mitchell MBA Msc

### Executive Summary

Using US Federal Reserve Open Market Committee public communications, we can forecast with approximately 65% accuracy the following 6-month interest rate sentiment.  This is based on using FOMC communications since 1960, and allowing the model to be trained on a random 75% training sample.  Removing portions of data based on date significantly reduces future model accuracy, as the features of the communications changes over time, and the economic/political environment changes over time.  When used with caution, an NLP model using FOMC communications as features can be a useful supplemental tool for interest rate forecasting.


### Problem Statement

Can NLP be used to forecast US rate changes with a useful level of accuracy?

### Requirements
Python 3.6+<br>
selenium<br>
fredapi<br>
pdfminer<br>
scikit-learn<br>
gensim<br>
nltk<br>
xgboost<br>
pyLDAvis<br>

### Data sources
[US Federal Reserve FRED Database](https://fred.stlouisfed.org/)<br>
[US FOMC](https://fraser.stlouisfed.org/title/677)
