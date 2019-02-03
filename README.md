# Disaster Response Pipeline Project
Create  a machine learning pipeline to categorize real messages send during disaster events.
Pipeline is used to categorize these events so that messages can be sent to an appropriate disaster
relief agency. This project includes a web app where an emergency worker can input
a new message and get classification results in several categories. The web app will aso
display visualizations of the data.

### Table of Contents

1. [Quick start](#start)
2. [What's included](#files)
3. [Licensing, Authors, and Acknowledgements](#licensing)

### 1. Quick start

---
1.1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

1.2. Run the following command in the app's directory to run your web app.
    `python run.py`

1.3. Go to http://0.0.0.0:3001/


### 2. What's included <a name="files"></a>

---
```
Disaster-Response-Pipelines/
└──.gitignore
├── app/
│   ├── run.py
│   └── templates/
│       ├── go.html
│       └── master.html
├── data/
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   └── process_data.py
├── LICENSE
├── models/
│   └── train_classifier.py
├── README.md
└── utilities/
    └── print_file_structure.py
```

### 3. Licensing, Authors, and Acknowledgements <a name='licensing'></a>
---
Must give credit to [Figure-eight](https://www.figure-eight.com/) for the data and   
[Udacity](https://www.udacity.com/courses/all) for creating a beautiful learning experience.  
Find the Licensing for the data and other descriptive information from [Figure-eight]
(https://www.figure-eight.com/dataset/combined-disaster-response-data/).


