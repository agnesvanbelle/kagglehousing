## Kaggle housing

###  Main appraoch
The file `src/main/main.py` contains some methods used to do grid search, cross-validation, training a final model and getting predictions on a test set. They should be the starting point to understand how the model was learnt.

### How to run: instructions

A REST-based API was made using Flask.

You need Python 3 (the code was made with Python 3.6.5).

Go to root directory of the project.  
It is recommended to create and start a virtual environment:  
```
python3 -m virtualenv env
source env/bin/activate  
```
Now install the packages:  
```
pip install -r requirements.txt
```
Go to the src directory:  
```
cd src
```
Start the server:  
```
PYTHONPATH=. python service/app.py
```
You should see a message saying that he server should be running at  `http://127.0.0.1:5000/`.  
You can now via your browser open that URL and get a prediction via the
interest_prediction endpoint.  
The parameters of that endpoint are:  
* bathrooms, integer
* bedrooms, integer
* building_id, string
* description, string
* feature, string (can be multiple)
* longitude, float
* latitude, float
* manager_id, string
* photo, string (link) (can be multiple)
* price, integer
* created, string using date format: "yyyy-mm-dd hh:mm:ss" or "yyyy-mm-dd"  

Example request:
```
http://127.0.0.1:5000/interest_prediction?bedrooms=1
&bathrooms=5
&latitude=40.7&longitude=-73.9425
&price=2000
&feature=a&feature=b&feature=c&feature=d
&description=%22a%20b%20c%20d%20e%20f%20g%22
&photo=b&photo=b&photo=c&photo=d
&created=2016-08-09%2018:50:00  
```
