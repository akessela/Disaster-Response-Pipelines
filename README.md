# Disaster Response Pipeline Project


### Table of Contents

1. [Quick start](#start)
2. [What's included](#files)
3. [Results](#results)
4. [Licensing, Authors, and Acknowledgements](#licensing)

### Quick start

---
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### What's included <a name="files"></a>

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
