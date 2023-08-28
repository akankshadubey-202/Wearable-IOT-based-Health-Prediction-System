import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Function to calculate NEWS score based on given parameters
def calculate_news(respiratory_rate, spo2, heart_rate, temperature, high_bp):
    score = 0
    # Calculate NEWS score for each parameter
    if respiratory_rate <= 8 or respiratory_rate>=25:
        score += 3
    elif respiratory_rate >= 9 and respiratory_rate <= 11:
        score += 1
    elif respiratory_rate >= 21 and respiratory_rate <= 24:
        score += 2
    
    if spo2 <= 91:
        score += 3
    elif spo2 >= 92 and spo2 <= 93:
        score += 2
    elif spo2 >= 94 and spo2 <= 95:
        score += 1
    
   
    
    if heart_rate <= 40 or heart_rate >=131:
        score += 3
    elif heart_rate >= 41 and heart_rate <= 50:
        score += 1
    elif heart_rate >= 91 and heart_rate <= 110:
        score += 1
    elif heart_rate >= 111 and heart_rate <= 130:
        score += 2


    if temperature <= 35.0:
        score += 3
    elif temperature >= 35.1 and temperature <= 36.0:
        score += 1
    elif temperature >= 38.1 and temperature <= 39.0:
        score += 1
    elif temperature >= 39.1:
        score += 2



    if high_bp <= 90 or high_bp >=220:
        score += 3
    elif high_bp >= 91 and high_bp <= 100:
        score += 2
    elif high_bp >= 101 and high_bp <= 110:
        score += 1
    
    return score
# Function to assign labels based on NEWS Score
def assign_label(news_score):
    if news_score >= 0 and news_score <= 4:
        return 0
    elif news_score >= 5 and news_score <= 6:
        return 1
    else:
        return 2
# Generate synthetic patient data
n_patients = 10  # Change this to the desired number of patients
start_date = datetime(2023, 7, 19)
end_date = datetime(2023, 8, 31)
date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

for patient_id in range(1, n_patients + 1):
    patient_data = []
    for date in date_range:
        # Generate synthetic data for each parameter (respiratory_rate, spo2, heart_rate, etc.)
        respiratory_rate = random.randint(8, 30)
        spo2 = random.randint(90, 100)
        heart_rate = random.randint(40, 160)
        temperature = round(random.uniform(35.0, 40.0), 1)
        high_bp = random.randint(90, 250)
        low_bp = random.randint(60, 150)
        physical_activity = random.choice(['Running', 'Walking', 'Standing', 'Sleeping'])
        accelerometer_x = random.uniform(-2.0, 2.0)
        accelerometer_y = random.uniform(-2.0, 2.0)
        accelerometer_z = random.uniform(-2.0, 2.0)
        gyrometer_x = random.uniform(-200.0, 200.0)
        gyrometer_y = random.uniform(-200.0, 200.0)
        gyrometer_z = random.uniform(-200.0, 200.0)
        
        # Calculate NEWS score
        news_score = calculate_news(respiratory_rate, spo2, heart_rate, temperature, high_bp)

        # Assign label based on NEWS Score
        label = assign_label(news_score)
        
        # Append data to the patient_data list
        patient_data.append([patient_id, date, respiratory_rate, spo2, heart_rate, temperature, high_bp, low_bp,
                             physical_activity, accelerometer_x, accelerometer_y, accelerometer_z, gyrometer_x,
                             gyrometer_y, gyrometer_z,news_score,label])

    # Create a DataFrame for the patient's data
    columns = ['User_ID', 'Date', 'Respiratory_Rate', 'Spo2', 'Heart_Rate', 'Temperature', 'High_BP', 'Low_BP',
               'Physical_Activity', 'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'Gyrometer_X',
               'Gyrometer_Y', 'Gyrometer_Z','news_score','label']
    patient_df = pd.DataFrame(patient_data, columns=columns)

    # Save the patient's data to an individual CSV file
    file_name = f'patient_{patient_id}.csv'
    patient_df.to_csv(file_name, index=False)
