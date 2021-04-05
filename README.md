# Health-Insurance-Lead-Prediction---JOB-A-THON
Analytics Vidhya

# Problem Background

## Problem Statement -
Your Client FinMan is a financial services company that provides various financial services like loan, investment funds, insurance etc. to its customers. FinMan wishes to cross-sell health insurance to the existing customers who may or may not hold insurance policies with the company. The company recommend health insurance to it's customers based on their profile once these customers land on the website. Customers might browse the recommended health insurance policy and consequently fill up a form to apply. When these customers fill-up the form, their Response towards the policy is considered positive and they are classified as a lead.

Once these leads are acquired, the sales advisors approach them to convert and thus the company can sell proposed health insurance to these leads in a more efficient manner.

Now the company needs your help in building a model to predict whether the person will be interested in their proposed Health plan/policy given the information about:
Demographics (city, age, region etc.)
Information regarding holding policies of the customer
Recommended Policy Information

# About the Dataset
* Visit this link for more information:-https://datahack.analyticsvidhya.com/contest/job-a-thon/#ProblemStatement

>>># Data Info:
             Data columns (total 13 columns):
             Column                         Non-Null Count  Dtype  
          ---  ------                       --------------  -----  
          0   City_Code                     50882 non-null  object 
          1   Region_Code                   50882 non-null  int64  
          2   Accomodation_Type             50882 non-null  object 
          3   Reco_Insurance_Type           50882 non-null  object 
          4   Upper_Age                     50882 non-null  int64  
          5   Lower_Age                     50882 non-null  int64  
          6   Is_Spouse                     50882 non-null  object 
          7   Health Indicator              39191 non-null  object 
          8   Holding_Policy_Duration       30631 non-null  object 
          9   Holding_Policy_Type           30631 non-null  float64
          10  Reco_Policy_Cat               50882 non-null  int64  
          11  Reco_Policy_Premium           50882 non-null  float64
          12  Response                      50882 non-null  int64  
          dtypes: float64(2), int64(5), object(6)

>>># Data Descriptions
          	Region_Code	Upper_Age	Lower_Age   Holding_Policy_Type	      Reco_Policy_Cat	Reco_Policy_Premium	Response
      count	50882.00	50882.00	50882.00	30631.00	        50882.00	    50882.00	        50882.00
      mean	1732.79	        44.86	        42.74	        2.44	                15.12	            14183.95	        0.24
      std	1424.08	        17.31	        17.32	        1.03	                6.34	            6590.07	        0.43
      min	1.00	        18.00	        16.00	        1.00	                1.00	            2280.00	        0.00
      25%	523.00	        28.00	        27.00	        1.00	                12.00	            9248.00	        0.00
      50%	1391.00	        44.00	        40.00	        3.00	                17.00	            13178.00	        0.00
      75%	2667.00	        59.00	        57.00	        3.00	                20.00	            18096.00	        0.00
      max	6194.00	        75.00	        75.00	        4.00	                22.00	            43350.40	        1.00
      
# Steps used during the project
1) Exploratary Data Analysis- 
  - Correlation with Target Variable
  - Countplot to know the Target Variables distribution
  - Stacked Bar plot to know the count of every Categorical Feature with respect to the Target Feature
  
 2) Preprocessing and Feature Engineering -
  - Dealing with different Date features by segregating the day, month and year into seperate columns
  - Filling missing values
  - Adding new Features to the Dataset
  - Label and One hot Encoding on the categorical columns
  
  3) Applying models-
  - The problem is a Classfication problem but instead of only predicting 0 and 1, we have to predict the probability of it. The Threshold is not changed thus indicating the default number of 0.5 (ie probability below 0.5 is represented as 0 and above 0.5 is represented as 1) 
  - Pycaret was initially applied to know the best classifiers.
  - Various models were used on the above data and even Stacking Ensemble was used to get the best outcome
  - The best two performing were the Catboost Classifier and the Stacking Ensemble Classification Method
  
  4) Evaluation Metric-
   
   - Model performance was evaluated on the basis of the probability of favourable outcome and the metric to judge it was Accuracy score.
   - Catboost Classifier gives best accuracy as compared to other models.
