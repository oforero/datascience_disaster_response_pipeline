# Disaster Response Pipeline Project

This projects uses the a Multiple Category Classification Model with a Neural Network as underlying model.

It mark each message as belogging to none, one, or many of the following message categories:

- Related
- Request
- Offer
- Aid Related
- Medical Help
- Medical Products
- Search And Rescue
- Security
- Military
- Child Alone
- Water
- Food
- Shelter
- Clothing
- Money
- Missing People
- Refugees
- Death
- Other Aid
- Infrastructure Related
- Transport
- Buildings
- Electricity
- Tools
- Hospitals
- Shops
- Aid Centers
- Other Infrastructure
- Weather Related
- Floods
- Storm
- Fire
- Earthquake
- Cold
- Other Weather

# Project Folder Structure

    disaster_response_pipeline (Workspace Root)
        |-- app
            |-- templates
                    |-- go.html
                    |-- master.html
            |-- run.py
        |-- data
            |-- disaster_message.csv
            |-- disaster_categories.csv
            |-- DisasterResponse.db
            |-- process_data.py
        |-- models
            |-- classifier.pkl
            |-- train_classifier.py
            |-- evaluate_classifier.py
        |-- README.md

# The Model

## Current Model Performance

Classification Report:
                         precision    recall  f1-score   support

               related       0.96      0.96      0.96      3998
               request       0.90      0.85      0.87       891
                 offer       0.00      0.00      0.00        24
           aid_related       0.90      0.91      0.91      2164
          medical_help       0.92      0.78      0.85       435
      medical_products       0.91      0.79      0.85       279
     search_and_rescue       0.95      0.67      0.78       136
              security       0.95      0.19      0.31        96
              military       0.95      0.82      0.88       158
                 water       0.93      0.88      0.90       335
                  food       0.95      0.91      0.93       584
               shelter       0.92      0.85      0.88       468
              clothing       0.97      0.56      0.71        70
                 money       0.88      0.65      0.75       112
        missing_people       0.95      0.33      0.49        63
              refugees       0.93      0.82      0.88       170
                 death       0.95      0.81      0.87       247
             other_aid       0.87      0.77      0.82       692
infrastructure_related       0.91      0.73      0.81       336
             transport       0.93      0.73      0.82       235
             buildings       0.96      0.83      0.89       269
           electricity       0.93      0.65      0.77       115
                 tools       0.00      0.00      0.00        35
             hospitals       0.94      0.31      0.46        52
                 shops       0.00      0.00      0.00        25
           aid_centers       0.93      0.20      0.33        64
  other_infrastructure       0.92      0.68      0.78       225
       weather_related       0.94      0.91      0.92      1472
                floods       0.94      0.85      0.89       431
                 storm       0.94      0.90      0.92       479
                  fire       0.92      0.66      0.77        53
            earthquake       0.95      0.90      0.92       515
                  cold       0.92      0.80      0.86       104
         other_weather       0.91      0.77      0.83       267
         direct_report       0.88      0.80      0.84      1010

             micro avg       0.93      0.85      0.89     16609
             macro avg       0.85      0.66      0.73     16609
          weighted avg       0.92      0.85      0.88     16609
           samples avg       0.70      0.67      0.67     16609

## Training Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
