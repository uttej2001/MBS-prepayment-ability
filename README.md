# **Project Description**

## **Name: Mortgage Based Payment Ability** <br><br>

### <b> Description of the Project </b><br><br>

This project specifically focuses on finding an alternative method to predict mortgage prepayment risk of residential mortgage loans by using different machine learning techniques.

We have used: Freddie Mac’s Single Family Loan- Level Dataset

This dataset originally contains 95 features divided as Original data and Performance data.

### <b> Overall flow of the project </b>
 
![image](https://user-images.githubusercontent.com/57294017/153746791-d26f11f7-3cae-4286-9986-2335940da9c7.png)


In the dataset we make the use of 27datacolumns.

**The description of the dataset can be found in the following document:
http://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf**


Before performing the EDA I want to drop four columns that seems really useless and doing nothing except for increasing the dimension of dataset.

These columns are:

'FIRST_PAYMENT_DATE',
'MATURITY_DATE',
'LOAN_SEQUENCE_NUMBER',
'METROPOLITAN_STATISTICAL_AREA',
'OCCUPANCY_STATUS',
'CHANNEL',
'PRODUCT_TYPE',
'PROPERTY_STATE',
'PROPERTY_TYPE',
'NUMBER_OF_BORROWERS',
'LOAN_PURPOSE','PREPAYMENT_PENALTY_MORTGAGE_FLAG',
'SELLER_NAME',
'SERVICER_NAME',
'POSTAL_CODE'

```python

mortgage_data.drop(['FIRST_PAYMENT_DATE','MATURITY_DATE','LOAN_SEQUENCE_NUMBER','METROPOLITAN_STATISTICAL_AREA','OCCUPANCY_STATUS','CHANNEL','PRODUCT_TYPE','PROPERTY_STATE','PROPERTY_TYPE','NUMBER_OF_BORROWERS','LOAN_PURPOSE','PREPAYMENT_PENALTY_MORTGAGE_FLAG',
'SELLER_NAME',
'SERVICER_NAME',
'POSTAL_CODE'], axis = 1,inplace = True)

```

## **Adding two new features to the dataset**

Here we introduced two new feature in the dataset named **'LTV_range'** and **'CreditScore_range'** derieved from **'ORIGINAL_LOAN_TO_VALUE'** and **'CREDIT_SCORE'** respectively.

In case of "ORIGINAL_LOAN_TO_VALUE" we can see that it ranges from 0 to 100. Now we will divide these continuous value into three parts and they are 
* Low(0-50), 
* medium(50-80) and 
* high(above 80)

In case of 'CREDIT_SCORE the minimum value for credit score is 300 and the maximum value is almost 850. So let's bin the value with the binning difference of 200. So 300-500,500-700,700-800.

* 300-500 = 1 [ Lowest value for credit score ]
* 500-700 = 2 [ medium value for credit score ]
* 700-900 = 3 [ Highest valye for credirt score ]

# **Data Analysis Phase [EDA]**

In this phase we will analyze the data to find out:

* Missing Values
Almost 3% of data is missing in our dataset.

* All the Continuous Varibles
The list of continuous feature are: 

['CREDIT_SCORE', 'MORTGAGE_INSURANCE_PERCENTAGE', 'ORIGINAL_COMBINED_LOAN_TO_VALUE', 'ORIGINAL_DEBT_TO_INCOME_RATIO', 'ORIGINAL_UPB', 'ORIGINAL_LOAN_TO_VALUE', 'ORIGINAL_INTEREST_RATE', 'ORIGINAL_LOAN_TERM']

 The continuous features are :  8
 
* Distribution of the continuous Variables

![image](https://user-images.githubusercontent.com/57294017/150626788-2ce9814c-07da-48ec-9ee1-eafa2ff08bcf.png)

![image](https://user-images.githubusercontent.com/57294017/150626807-befab554-e175-42d7-9e41-5c91b614b77d.png)

![image](https://user-images.githubusercontent.com/57294017/150626821-832c5483-df03-4566-beb6-47de2d366b2c.png)

![image](https://user-images.githubusercontent.com/57294017/150626828-1eca84ae-13f5-4f33-afbb-35bd6aeb4541.png)

![image](https://user-images.githubusercontent.com/57294017/150626842-ffcd395f-b707-4351-a33f-09728e8f03db.png)

![image](https://user-images.githubusercontent.com/57294017/150626851-c8013091-2a8d-4a3a-a973-e0dfd1cc44dc.png)

![image](https://user-images.githubusercontent.com/57294017/150626869-b6e8d186-bc3c-4a3b-b581-14fb603876c2.png)

![image](https://user-images.githubusercontent.com/57294017/150626879-6fdcd27b-89b0-43d3-9578-ce0286b01eb8.png)

#### <b> Analysis :</b>

Let's decode the distribution of these 8 continuous features one by one:

From above observation:

* **Credit score(left skewed)**
The credit score feature doesn't follow **gaussian distribution** rather it is slightly **leftskewed**. From the distribution plot we can say that the credit score value of many borrowers mainly lies in between the range of 500-800.

* **MORTGAGE_INSURANCE_PERCENTAGE**
The Mortgage insurance percentage feature values are very sparsely distributed. Most of the borrowers Mortage insurance percentage has % slightly more than 0%. Besides that the percentage distribution doesn't exceed from more than 30-35%.

* **ORIGINAL_COMBINED_LOAN_TO_VALUE**
In this feature the combined loan to value of the borrowers is mainly distributed between 25% to 100% most of the borrowers have CLTV around 75%.


* **ORIGINAL_DEBT_TO_INCOME_RATIO**
The values in this feature are very well distributed and thus follows the normal distribution the ratio value ranges from 0 to 70.

* **ORIGINAL_UPB**
This values in this features are mostly right skewed. This feature is actually UPB of the mortgage on the note date. Most of the value lies in between 0 to 300000.

* **ORIGINAL_LOAN_TO_VALUE**
This feature also doesn't follow a normal distribution. The value range form 0 t0 100% and most of the borrowers has around 80 to 85% of LTV.

* **ORIGINAL_INTEREST_RATE**
This feature somewhat follows gaussian distribution and the Interest rate mostly lies in between 6-9%.

* **ORIGINAL_LOAN_TERM**
This feature is obtained by subtracting the Maturity Date from the first payment date i.e. 

Original_loan_term = (Loan
Maturity Date (MM/YY)
– Loan First Payment
Date (MM/YY) + 1

The dataset only contains the loan tern in between 300 to 420 and rest of the loan terms are all excluded from the dataset.


* Categorical varibles

![image](https://user-images.githubusercontent.com/57294017/150626936-43bfefd0-fce9-46aa-a09a-8a675b990ded.png)

#### **Analysis:**

The **'FIRST_TIME_HOMEBUYER_FLAG'** feature is a categorical feature with two categories either the borrowers are the first time homebuyer or not. This feature also has many missing value that depicts that some of the borrowers record are not available or they are not aplicable.

From the above count plot we can say that most of the borrowers are not first time homebuyer.

* Observing the target variable

![image](https://user-images.githubusercontent.com/57294017/150626687-f272c8b8-217b-4668-bf4b-843284e591f6.png)


The prepaid percentage is :
 96.11846354098978
The non-prepaid percentage is :
 3.8275512509572374

 #### <b>Analysis: </b> 

We can clearly see the data imbalance here. As the number of prepaid is extremly higher than the non-prepaid data. i.e.** prepaid = 480724**, **non_prepaid = 19413**.**96%** of the data is prepaid and only **4%** of the data here is not-prepaid.

**Imbalalance data** can sometime affect our models performance and can bring biasness in our models performance. For e.g. In case of our data if we didn't handle the imbalance data then we might end up building a bias model with a lot of false positive value. That's quiet a problem. So we need to handle the imbalnce dataset using 

1. Undersampling of majority class label.
2. Oversampling of minority class label
3. Using SMOTE Technique which adds synthetic datapoints to our minority class label in order to balance the number of data points in each classes.
* Outliers
* Relationship between Independent and dependent feature


### <b> Data preprocessing </b><br><br>

We will:

1. Handle missing values 
2. Do data cleaning
3. Handle categorical values
4. Transform data( scale, normalize the data, dimensionality reduction

### Handling Missing Values

Imputation is a very important way to manage the null values in the dataset. But sometimes imputation might not be always useful specially replacing the values in the feature with it's mean,median or mode value or frequently used value can be sometimes misleading. If you have less number of value in your dataset you can simply drop the rows with all the null values and see what actually happens. 

In case of our dataset we have 7 columns with missing rows.


* CREDIT_SCORE                         2711
* FIRST_TIME_HOMEBUYER_FLAG          130559
* MORTGAGE_INSURANCE_PERCENTAGE       51048
* NUMBER_OF_UNITS                         3
* ORIGINAL_COMBINED_LOAN_TO_VALUE        13
* ORIGINAL_DEBT_TO_INCOME_RATIO       14929
* ORIGINAL_LOAN_TO_VALUE                  9


Here, **FIRST_TIME_HOMEBUYER_FLAG** feature has maximum number of values and If we drop the rows which contains these null values we might lose lot of data. So, we need to find a decent way to manage the null value in these 2 features.

**SOLUTION:**

In case of **FIRST_TIME_HOMEBUYER_FLAG** I will replace all the null values with "others" which **means either the value is not available or the borrowers are not applicable**.

In case of remaining columns with null values we will simply **drop the rows containing these null values**.

```python
# Firstly filling all the null values in FIRST_TIME_HOMEBUYER_FLAG with "O" which means others

mortgage_data_copy["FIRST_TIME_HOMEBUYER_FLAG"].fillna("O", inplace = True)
```

```python
# Let's drop all the rows with the null values

mortgage_data_copy.dropna(inplace = True)
```

### <b> Handaling categorical variables </b>

We know that our machine learning model only can work on the numerical value so it's really important to convert the categorical value in our dataset into some understandable numerical form so that it can be used for the training purpose.

We can follow different methods for handaling the categorical values in our dataset:

1. Label Encoding
2. One hot encoder

------------------------------------------------------------------------------

After dropping the columns in our dataset we have just 2 categorical features remaining after dropping i.e. 

* FIRST_TIME_HOMEBUYER_FLAG,
* LTV_range


We will use one_hot encoding technique to encode our categorical feature. 

If we have categorical variables containing many multiple labels or high cardinality,then by using one hot encoding, we will expand the feature space dramatically.

One approach that is heavily used to solve this problem is to replace each label of the categorical variable by the count, this is the amount of times each label appears in the dataset. Or the frequency, this is the percentage of observations within that category. The 2 are equivalent.

## <b> Finally performing the FEATURE SCALING </b> - Using StandardScalar

Feature Scaling is a technique to standardize the independent features present in the data in a fixed range.

If feature scaling is not done, then a machine learning algorithm tends to weigh greater values, higher and consider smaller values as the lower values, regardless of the unit of the values.

In our dataset we have features like 'METROPOLITAN_STATISTICAL_AREA' with huge range of value whereas 'MORTGAGE_INSURANCE_PERCENTAGE','ORIGINAL_LOAN_TO_VALUE' and so on has less range of value and in that case we require scaling of the data.

Here we will use <b>Standard Scalar</b> to scale our data. Minmax scaling scales the data between 0 and 1 using formula 

Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

Standardize features by removing the mean and scaling to unit variance.

The standard score of a sample x is calculated as:

<b> z = (x - u) / s </b>

where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

# **Model Building**

### <b> Description of the first Model </b><br><br>

We used **Logistic Regression** as the first algorithm to train our dataset.

After doing the necessary **EDA** and **Data Preprocessing** we splited our model into train test sets 

```python

X_train,X_valid,y_train,y_valid = train_test_split(X_encoded,y,test_size= 0.2,random_state=0)

```
After that we scaled the training and validation data using the **StandardScaler**.

We created a preliminary logisticRegression model using **k-fold cross validation** for validating our model and we used 'C' parameter as the regularization parameter. We also used **class-weight** parameter which actually didn't aid us much so later on removed it.

```python
# Model building using logistic regression
lr = LogisticRegression()

# Creating hyperparameter tuning for implementing cross validation
grid = {'C':10.0 ** np.arange(-2,3)}
cv = KFold(n_splits=5,shuffle= False,random_state = None)

clf = GridSearchCV(lr,grid,cv = cv,n_jobs = -1, scoring = 'accuracy')
clf.fit(x_train_std,y_train)

```
After training the model we tested the model on our validation data.

**We got a the accuracy of 96.5%**

### **Model's Performance Analysis:**

we used confusion matrix and classification report to analyze the performance of our model.

**Result of confusion matrix:**

![image](https://user-images.githubusercontent.com/57294017/150059943-c09689c2-6226-4ea1-baaa-c4c0f7bd85bf.png)


In case of confusion metric we can see that our model is correctly able to classify **81752(True positive**) and **1608(True Negative)** Value. But we have pretty decent number of **False Negative(953)** and **False positive(2048)** falsely classified by our model. **In our case we need to try to reduce the number of false positive because classifying a non-prepaid value as prepaid can result in more loss to the bank** than the otherway analogy. Also our data is highly imbalanced that could also be one reason why our model is performing that way. So we will also try balancing out the data set to see if will get some improvement.

From any company's point of view predicting prepaid when it is actually not is kind of more dangerous and we can see that our model has falsely classified a lot of unprepaid value as prepaid. So we also need to try to **reduce the false positive**.


**Result of classification Report**

![image](https://user-images.githubusercontent.com/57294017/150059991-382c65e2-b5ac-4268-b233-4bead46f0650.png)


From the report it's clear that the value of precision, recall and  f1 score is quite low for the class label 0 whereas for class label 1 it's pretty high.

We also got actually a good ROC-AUC score of about is **0.89**. But as this a highly imbalanced dataset there is high chance that this score is a bias score.

### **ROC curve for to see the realationship of TPR and FPR**

![image](https://user-images.githubusercontent.com/57294017/150061125-74494f7c-e125-451f-ba2a-16cefb8dce96.png)

So from the above graph we can say that if we need only care about the true +ve positive rate then we can use the threshold of 0.5 but if we want to reduce the true +ve rate we can also change the threshold value to the value something bigger than 0.5.

### **Balancing the imbalanced data and performing the training**<br>

We used the cocept of random oversampling to increase the number of minority class. Random oversampling involves randomly selecting examples from the minority class, with replacement, and adding them to the training dataset. 


``` python
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

os = RandomOverSampler(0.5)
X_train_ns,y_train_ns = os.fit_resample(x_train_std,y_train)
print("The number of classes before fit{}".format(Counter(y_train)))
print("The number of classes after fit{}".format(Counter(y_train_ns)))

```

**Result:**

```
The number of classes before fitCounter({True: 330873, False: 14569})
The number of classes after fitCounter({True: 330873, False: 165436})

```

Then we applied the similar model training approach like above and perform everything from the beginning.

**We got a the accuracy of 95.7%**


### **Performance Analysis of the model after balancing the dataset using the technique of random oversampler**

![image](https://user-images.githubusercontent.com/57294017/150060065-aed877e6-d9d7-4685-b2bd-b2258b695a09.png)

![image](https://user-images.githubusercontent.com/57294017/150060111-6a2140f8-8f2b-4bfe-a839-ee5d64246aac.png)

**Analysis:** After balancing the data we could see slight increase in the score of recall and f1 score and we have lesser number of false positive as well.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
We also used the technique of SMOTE in order to perform the data balance. But the performance of this technique was not as good as random oversampling.

```python

from imblearn.combine import SMOTETomek


os = SMOTETomek(0.75)

# Taking only 50000 points from the training dataset
X_train_ns1 = x_train_std[0:50000]
y_train_ns1 = y_train[0:50000]

X_train_ns1,y_train_ns1 = os.fit_resample(X_train_ns1,y_train_ns1)
print("The number of classes before fit{}".format(Counter(y_train)))
print("The number of classes after fit{}".format(Counter(y_train_ns1)))

```

Then we applied the similar model training approach like above and perform everything from the beginning.

**We got a the accuracy of 90%**

### **Performance Analysis of the model after balancing the dataset using SMOTE**

![image](https://user-images.githubusercontent.com/57294017/150060680-4fca6a93-279c-4da3-a3cc-7360d9b6ae58.png)

From the analysis we see that we have incresed the decresed the number of true +ve. The recall has incresed but their is a significant decrese in the value of precision which resulted in small f1 score less than 50.


**Random Oversamping was better than SMOTE technique because:**

* It's computational speed was exceptionally good than SMOTE.
* We were also able to get good recall and were able to decrease the false positive and has a good f1-score using **random over sampler** but in case of **SMOTE** the result for f1 score is worst.

# **Web Application using Flask framework**
### <u> Description of the web application </u>
The web application uses simple techniques for it's implementaion.

* <b> HTML & CSS</b> for the fromend part.
* <b> Flask </b> is used in backend.


### Workflow of the project

The user inputs necessary data into the form like structure and once click the predict button the result is displayed on the screen itself. 

Logic behind:

The main file that we need o focus on is app.py. 

STEP 1:  
We have done necessary imports of the library that we need to implement our logic.

```python
from distutils.log import debug
import numpy as np
from flask import Flask,render_template,request
from sklearn.preprocessing import StandardScaler
import joblib

```


STEP 2:  
After that we read the pickled file using the Joblib library.

STEP 3:

Then we define a function named predict prepayment where we basically take in the input from the user and then convert the input into the float format and then scale down the value and finally call the <b>.predict</b>  function which does the prediction job.


```python
# Demo

def predict():

    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    final_features_scaled = standard_to.fit_transform(final_features)
    prediction = model.predict(final_features_scaled)

    if (prediction == True):
        result = "It is a Prepaid condition"
    
    elif (prediction == False):
        result = "It is Non-prepaid condition"

    else:
        result = "Not available"

    return render_template("index.html", result= result)


```

### Screenshot of the project

#### UI of the project

![image](https://user-images.githubusercontent.com/57294017/152632487-700ded30-d8a4-4fbb-970a-621e81188562.png)


![image](https://user-images.githubusercontent.com/57294017/152632519-f5a1abcb-148a-4d1b-b7d9-dce301012c51.png)

![image](https://user-images.githubusercontent.com/72940291/162268785-8b31e7c1-37c7-4e75-b8ed-2370eb2c30af.png)


### Result for the prediction

For the prepaid condition
![image](https://user-images.githubusercontent.com/57294017/152645973-e8b93c84-473b-4f70-a551-d0b9072e196b.png)
![image](https://user-images.githubusercontent.com/57294017/152646048-98075aac-3961-4d0f-a997-ef2075f79cf7.png)

For the non-prepaid condition
![image](https://user-images.githubusercontent.com/57294017/152646364-cb1398bb-6087-4ace-8a4a-51b8c92541a3.png)


# **Deployment Link for Model Using Logistic Regression**
 https://ml-mortgageprepayment.herokuapp.com/






