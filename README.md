# Predicting Excitement at DonorsChoose.org
[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://sofiaaparicio.github.io/)  [![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/) 

This repository contains my solution to a Kaggle competion entitled [KDD Cup 2014 - Predicting Excitement at DonorsChoose.org][kaggle_comp] Predict funding requests that deserve an A+. The main goal of this project was to predict if a determined project was exciting to the business or not, based on several features.

## Overview

These were the steps I took:

1 - I did a data exploration ([EDA.ipynb](/datasets/EDA.ipynb/)) to better understand the available data and to perceive how should be treated, alongside the models that could be applied;

2 - After the EDA, I built a pipeline to guide my work;

3 - Preprocessed the datasets([data_preprocessing.ipynb](/datasets/data_preprocessing.ipynb/))  and originated one input dataset;

4 - Then, applied a feature transformation, alongside generating a model and predicting the results for the competition ([Feature_Transformation_Predictions.ipynb](/Feature_Transformation_Predictions.ipynb/));


## Pipeline

![](/images/pipeline_DonorsChoose.png)

## Data



Originally the input data was composed by three datasets that had training information (essays.csv, projects.csv, resources.csv). I did not use the other two available datasets has input data because they only hold information about or the training set alone (donations.csv, outcomes.csv) and could not be used as input for the test set aswell.
 
The data preprocessing merges the three datasets (essays.csv, projects.csv, resources.csv) into one input dataset, after it separates the dataset into the input training and input test. 

The final dataset has the following columns:

* projectid - project's unique identifier 
* title_v - (derived from essays.csv column title) classify title sentiment dimension Valence
* title_a - (derived from essays.csv column title)  classify title sentiment dimension Arousal
* numb_words_short_description - (derived from essays.csv column short_description) number of words in short_description
* most_common_short_description - (derived from essays.csv column short_description) number of common words in short_description
* numb_words_essay - (derived from essays.csv column essay) number of words in essay
* teacher_acctid -  (derived from projects.csv) project's unique identifier *
* schoolid - (derived from projects.csv) school's unique identifier *
* school_latitude - (derived from projects.csv) school's latitude 
* school_longitude - (derived from projects.csv) school's latitude 
* school_city - (derived from projects.csv) school's city *
* school_state - (derived from projects.csv) school's state *
* school_zip - (derived from projects.csv) school's zipcode
* school_metro - (derived from projects.csv) school's metro *
* school_charter - (derived from projects.csv) converted to bool
* school_magnet - (derived from projects.csv) converted to bool
* school_year_round - (derived from projects.csv) converted to bool
* school_nlns - (derived from projects.csv) converted to bool
* school_kipp - (derived from projects.csv) converted to bool
* school_charter_ready_promise - (derived from projects.csv) converted to bool
* teacher_teach_for_america - (derived from projects.csv) converted to bool
* teacher_ny_teaching_fellow - (derived from projects.csv) converted to bool
* eligible_double_your_impact_match - (derived from projects.csv) converted to bool
* eligible_almost_home_match - (derived from projects.csv) converted to bool
* teacher_prefix - (derived from projects.csv) teacher's prefix *
* primary_focus_subject - (derived from projects.csv) *
* primary_focus_area - (derived from projects.csv) *
* secondary_focus_subject - (derived from projects.csv) *
* secondary_focus_area - (derived from projects.csv) *
* resource_type - (derived from projects.csv) *
* poverty_level - (derived from projects.csv) #
* grade_level - (derived from projects.csv) #
* fulfillment_labor_materials - (derived from projects.csv)
* total_price_excluding_optional_support - (derived from projects.csv) 
* total_price_including_optional_support - (derived from projects.csv) 
* students_reached - (derived from projects.csv) 
* year_posted - (derived from projects.csv column date_posted)
* month_posted - (derived from projects.csv column date_posted)
* day_posted - (derived from projects.csv column date_posted)
* item_quantity - (derived from resources.csv) 
* item_total_quantity - (derived from resources.csv) 
* vendor_name - (derived from resources.csv) *


\* To do categorical non ordinal encoding, Target encoding, on [data_preprocessing.ipynb](/datasets/data_preprocessing.ipynb/)

\# To do categorical ordinal encoding on [data_preprocessing.ipynb](/datasets/data_preprocessing.ipynb/)


From some columns it was necessary to treat the NaN values. 
- school_zip - zipcode was deduced from latitude and longitude values. 
- school_metro, primary_focus_subject, primary_focus_area - they where replaced with the value 'Other'.
- secondary_focus_subject - they were replaced by the primary_focus_subject
- secondary_focus_area - they were replaced by the primary_focus_area
- all nan on the columns of the essays.csv were replaced by empty strings

## Feature Transformation and Predictions

#### 1 - Apply Feature Scaling

##### 1.1 - Encode Categorical Data
It is necessary to encode the categorical features every time we using diferent chunks of the dataset has train and test data.
Used target encoding for non ordinal categorical data, primarily proposed on an article[1] an then coded into a library compatible to sklearn.

##### 1.2 - Scale data
When the data is used on a machine learning model, the data needs to be scaled. Feature scaling relies on uniformizing the data. There are two main techniques of scaling the data: Normalizing and Standardization. I used Standardizations instead of Normalization because in this dataset there are a several column with outliers. Since Normalization has a bounding range (meaning, it uses the minimum and maximum values to normalize), giving highly importance to outlier.

#### 2 - Dimensionality Reduction
To avoid overfitting and reduce the complexity of the machine learning model, it is necessary to apply dimensionality reduction to then dataset. To do so, PCA will be used.

![](/images/pca.png)

It was possible to conclude that it is necessary 27 components to explain at least 80% of the variability of the original dataset. And 36 components to explain at least 95% .

#### 3 - Treating Imbalanced Data

In this section I use a simple Decision Tree Classifier to observe the differences between the several methods to treat imbalanced data.

Firts, it was calculated a simple prediction without treating data unbalance. Then used two oversampling methods (Random Over Sampler and SMOTE). One undersampling method (Random Under Sampler). The results are on the table bellow.

 Metrics | Not balanced   | Random Oversampling  | SMOTE | Random Undersampling 
---------|----------------|----------------------|-------|----------------------
 F1 Score with validation data | 0.780         |  0.796               |0.747  |0.596 
  Balanced Accuracy with validation data | 0.780| 0.503                |0.499  |0.590
 ROC with test data | 0.499 | 0.496 | 0.507 | 0.501

Observing the table, it is possible to conclude that random undersampling provides worst F1-score and Balanced Accuracy values than the rest of the methods of oversampling and even the not balanced data. However, it was the second best result when submited to kaggle, only the SMOTE shows an higher ROC result.

After I tried ensemble methods. These are models combined with methods that generate data undersampling. Khoshgoftaar et al.[2] defends the use of Boosting and Bagging for imbalanced data. Defending that, if the data is clean and imbalanced, both methods tend to perform well. Defending that Balanced Bagging tends to performe very well, compared to other methods. And also stated that RUSBoost outperformes other booting method, such as SMOTEBoost, so I will test this method.

Chen et al.[3] tested inbalanced data with Weighted Random Forest and Balanced Random Forestes, observing that the Balanced Random Forests provided better results

 The results of the experiments are on the table bellow.

 Metrics | Balanced Random Forest Classifier   | Balanced Bagging Classifier | RUSBoost Classifier 
---------|----------------|----------------------|-------
F1 Score with validation data | 0.676        |  0.762       | 0.627  
Balanced Accuracy with validation data | 0.501 | 0.503      | 0.507 
ROC with test data | 0.498 | 0.501 | 0.510 

As final model, I combined three models into a Voting classifier: RUSBoost Classifier, Random Forest Classifier and Gradient Boosting Classifier. It returned an F1 score of 0.760 and a Balanced Accuracy score for 0.502. After submissin, the roc score was 0.50416. 


## Geting Started
    pip install requirements.txt

## Resources

[1]Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. ACM SIGKDD Explorations Newsletter, 3(1), 27-32.
ISO 690	

[2]Khoshgoftaar, T. M., Van Hulse, J., & Napolitano, A. (2010). Comparing boosting and bagging techniques with noisy and imbalanced data. IEEE Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans, 41(3), 552-568.

[3]Chen, C., Liaw, A., & Breiman, L. (2004). Using random forest to learn imbalanced data. University of California, Berkeley, 110(1-12), 24.

[kaggle_comp]: <https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/overview>