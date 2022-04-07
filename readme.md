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
![image](https://user-images.githubusercontent.com/72940291/162266353-c6df66b5-e07e-45f9-8564-44f8d6f2ce60.png)
![image](https://user-images.githubusercontent.com/72940291/162266432-ce356126-030d-4c26-8d59-4abb91de8c6f.png)

### Result for the prediction

![image](https://user-images.githubusercontent.com/57294017/152645973-e8b93c84-473b-4f70-a551-d0b9072e196b.png)

For the prepaid condition

![image](https://user-images.githubusercontent.com/57294017/152646048-98075aac-3961-4d0f-a997-ef2075f79cf7.png)

For the non-prepaid condition

![image](https://user-images.githubusercontent.com/57294017/152646364-cb1398bb-6087-4ace-8a4a-51b8c92541a3.png)


### Deployment Link for Model Using Logistic Regression:


