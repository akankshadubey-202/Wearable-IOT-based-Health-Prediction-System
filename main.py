import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash,  jsonify,Response
import xgboost as xgb
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
import csv
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from flask import send_file


app = Flask(__name__)
app.secret_key = "r@nd0mSk_1"


# Load the model
model1 = joblib.load('models/heart_model.pkl')  # Replace with the actual path to your model
model2 = joblib.load('models/bp_model.pkl')
model3 = joblib.load('models/resp_model.pkl')
model4 = joblib.load('models/spo2_model.pkl')
model5 = joblib.load('models/temp_model.pkl')

def get_db_connection():
    conn = sqlite3.connect('healthdb.db',timeout=1)
    return conn

# Helper functions for user registration and login
def register_user_to_db(username, password):
    con = get_db_connection()
    cur = con.cursor()
    cur.execute('INSERT INTO users(username,password) values (?,?)', (username, password))
    con.commit()
    con.close()

def check_user(username, password):
    con = get_db_connection()
    cur = con.cursor()
    cur.execute('SELECT username, password FROM users WHERE username=? and password=?', (username, password))

    result = cur.fetchone()
    if result:
        return True
    else:
        return False
    
@app.route("/")
def indexes():
    return render_template('intro.html')
# Routes for user registration and login
@app.route("/start")
def index():
    return render_template('login.html')
@app.route("/home-temp")
def dashboard():
    return render_template('home.html')

@app.route('/register', methods=["POST", "GET"])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        register_user_to_db(username, password)
        return redirect(url_for('index'))
    else:
        return render_template('register.html')

@app.route('/login', methods=["POST", "GET"])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if check_user(username, password):
            session['username'] = username
            return redirect(url_for('show_counts'))
        else:
            return "Invalid username or password"
    else:
        return redirect(url_for('index'))

@app.route('/home', methods=['POST', "GET"])
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    else:
        return "Username or Password is wrong!"

# Route for user logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Create or connect to the SQLite database for user profiles

def get_user_id_from_username(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users_info WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user['id'] if user else None

# Define a function to generate PDF from user_info dictionary
# Define a function to generate PDF from user_info dictionary
def generate_pdf(user_info):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"User ID: {user_info['id']}", styles['Normal']))
    story.append(Paragraph(f"Username: {user_info['username']}", styles['Normal']))
    story.append(Paragraph(f"Name: {user_info['first_name']} {user_info['last_name']}", styles['Normal']))
    story.append(Paragraph(f"Age: {user_info['age']}", styles['Normal']))
    # Add other user details here

    doc.build(story)
    buffer.seek(0)

    return buffer
    
# Route to download user info as PDF
@app.route('/download-user-info-pdf/<int:user_id>')
def download_user_info_pdf(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users_info WHERE id = ?', (user_id,))
    user_info = cursor.fetchone()

    conn.close()

    if user_info:
        # Convert user_info (tuple) to a dictionary
        user_info_dict = {
            'id': user_info[0],
            'username': user_info[1],
            'first_name': user_info[2],
            'last_name': user_info[3],
            'age': user_info[4],
            # Add other fields as needed
        }

        pdf_buffer = generate_pdf(user_info_dict)

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f'user_info_{user_info[0]}.pdf',
            mimetype='application/pdf'
        )
    else:
        # Handle the case where user_id doesn't exist in the database
        flash('User not found', 'error')
        return redirect(url_for('dashboard'))

# Route for submission success page
@app.route('/submission-success/<user_id>')
def submission_success(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users_info WHERE id = ?', (int(user_id),))
    user_info = cursor.fetchone()

    conn.close()

    return render_template('success.html', user_info=user_info)
# Route for the user profile form
@app.route('/user-profile', methods=['GET', 'POST'])
def user_profile():
    if request.method == 'POST':
        action = request.form['action']

        if action == 'submit':
            # Process the form submission and store the data in the user_info table
            username = request.form['username']
            first_name = request.form['first']
            last_name = request.form['last']
            age = request.form['age']
            gender = request.form['gender']
            heartproblem = request.form.get('Hear_Problems', 'N')
            diabetes = request.form.get('diabetes', 'N')
            highbp = request.form.get('high_bp', 'N')
            lowbp = request.form.get('low_bp', 'N')
            asthma = request.form.get('asthma', 'N')
            obesity = request.form.get('obesity', 'N')
            none = request.form.get('none', 'N')

            conn = get_db_connection()
            cursor = conn.cursor()
            # Insert data into user_info table
            cursor.execute('''
                INSERT INTO users_info (username, first_name, last_name, age, gender, asthma, heartproblem, highbp, lowbp, obesity, diabetes, none)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (username, first_name, last_name, age, gender, asthma, heartproblem, highbp, lowbp, obesity, diabetes, none))
            
            conn.commit()
            # Retrieve the last inserted ID
            user_id = cursor.lastrowid
            conn.close()
            
            return redirect(url_for('submission_success', user_id=user_id))

        elif action == 'update':
            # Process the form update and update the data in the user_info table
            # Get the user_id based on the session's username
            username = session['username']
            
            username = request.form['username']
            first_name = request.form['first']
            last_name = request.form['last']
            age = request.form['age']
            gender = request.form['gender']
            heartproblem = request.form.get('Hear_Problems', 'N')
            diabetes = request.form.get('diabetes', 'N')
            highbp = request.form.get('high_bp', 'N')
            lowbp = request.form.get('low_bp', 'N')
            asthma = request.form.get('asthma', 'N')
            obesity = request.form.get('obesity', 'N')
            none = request.form.get('none', 'N')
            
            conn = get_db_connection()
            cursor = conn.cursor()
            # Update data in user_info table
            cursor.execute('''
                UPDATE users_info
                SET first_name=?, last_name=?, age=?, gender=?, asthma=?, heartproblem=?, highbp=?, lowbp=?, obesity=?, diabetes=?, none=?
                WHERE username=?
            ''', (first_name, last_name, age, gender, asthma, heartproblem, highbp, lowbp, obesity, diabetes, none, username))
            
            conn.commit()
            conn.close()

    # Handle the GET request to display the user profile form
    return render_template('user_profile.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'csv-file' not in request.files:
            flash('No CSV file uploaded.')
            return redirect(request.url)

        csv_file = request.files['csv-file']

        if csv_file.filename == '':
            flash('No selected file.')
            return redirect(request.url)

        if csv_file:
            #user_id = request.form['user-id']  # You should have a form field for the user's ID

            # Create a table name based on the user's ID (you can adjust this naming convention)
           # table_name = f'user_{user_id}_data'

            # Save the uploaded CSV file
            csv_filename = os.path.join('uploads', csv_file.filename)
            csv_file.save(csv_filename)

            # Read the CSV file and insert data into the dynamically created table
            with open(csv_filename, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the header row
                conn = get_db_connection()
                cursor = conn.cursor()

                # Create a new table for the user's data
                cursor.execute(f'''CREATE TABLE IF NOT EXISTS health_data (
            UserID INTEGER,
            Date DATE,
            Respiratory_Rate INTEGER,
            Spo2 INTEGER,
            Heart_Rate INTEGER,                
            Temperature REAL,
            High_BP REAL,
            Low_BP REAL,
            Physical_Activity TEXT,
            Accelerometer_X REAL,
            Accelerometer_Y REAL,
            Accelerometer_Z REAL,
            Gyrometer_X REAL,
            Gyrometer_Y REAL,
            Gyrometer_Z REAL,
            FOREIGN KEY (UserID) REFERENCES users_info(id)
                )''')

                # Insert data into the user's table
                for row in csv_reader:
                    cursor.execute(f"INSERT INTO health_data (UserID, Date, Respiratory_Rate, Spo2,Heart_Rate,Temperature, High_BP,Low_BP, Physical_Activity, Accelerometer_X, Accelerometer_Y, Accelerometer_Z, Gyrometer_X, Gyrometer_Y, Gyrometer_Z) VALUES (?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                   (row[0],row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14]))

                conn.commit()
                conn.close()

            flash('CSV file uploaded and data inserted into the database successfully.')
            return redirect(url_for('upload'))

    return render_template('upload.html')

def fetch_health_data(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT HeartRate, Date
        FROM health_data
        WHERE UserID = ?
    ''', (user_id,))
    data = cursor.fetchall()
    conn.close()
    return data

'''@app.route('/get_health_data')
def imp():
    return render_template('index.html')'''

@app.route('/get_health_data', methods=['GET'])
def get_health_data_route():
    user_id = request.args.get('user_id')
    if user_id:
        # Connect to the SQLite database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query the database for heart rate and date data based on user_id
        cursor.execute("SELECT  Date, Respiratory_Rate, Spo2,Heart_Rate,Temperature, High_BP,Low_BP FROM health_data WHERE UserID = ?", (user_id,))
        data = cursor.fetchall()

        # Close the database connection
        conn.close()

        date = [row[0] for row in data]  # Extract heart rates into a list
        Respiratory_Rate = [row[1] for row in data]  # Extract dates into a list
        spo2_levels = [row[2] for row in data]
        heart_rates= [row[3] for row in data]
        temp= [row[4] for row in data]
        ubp= [row[5] for row in data]
        lbp= [row[6] for row in data]
        result = {'HeartRates': heart_rates, 'Dates': date, 'Spo2' :spo2_levels,'Respiratory_Rate':Respiratory_Rate, 'UpperBP':ubp,'LowerBP':lbp,'BodyTemperature':temp}  # Store in a dictionary
        #return jsonify(result)
        return render_template('chart.html', dates=result["Dates"], heart_rates=result["HeartRates"],spo2_levels=result['Spo2'], Respiratory_Rate=result['Respiratory_Rate'],ubp=result['UpperBP'],lowbp=result['LowerBP'],temp=result['BodyTemperature'])
    else:
        return jsonify({'error': 'User ID is missing'})


#MACHINE LEARNING PREDICTION 



# Define functions to fetch health data and user information from the SQLite database
def fetch_health_data(user_id, date):
    # Establish a connection to the database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Execute a query to fetch health data
    cursor.execute(f"SELECT  Respiratory_Rate, Spo2,Heart_Rate,Temperature, High_BP,Low_BP FROM health_data WHERE UserID = ? AND Date = ?", (user_id, date))
    health_data = pd.DataFrame(cursor.fetchall(), columns=["Respiratory_Rate","Spo2" ,"Heart_Rate","Temperature","High_BP","Low_BP"])  # Adjust column names

    # Close the database connection
    conn.close()

    return health_data

def fetch_user_data(user_id):
    # Establish a connection to the database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Execute a query to fetch health data
    cursor.execute(f"SELECT username,first_name,last_name,age,gender FROM users_info WHERE id = ?", (user_id))
    user_info = pd.DataFrame(cursor.fetchall(), columns=["username","first_name","last_name","age","gender"])  # Adjust column names

    # Close the database connection
    conn.close()

    return user_info

# New route for SpO2 prediction form
@app.route("/predictions", methods=["GET"])
def prediction():
    return render_template("predictions.html")

# Route to handle SpO2 prediction
@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        user_id = request.form['user_id']
        date = request.form['date']
        # Fetch SpO2 data from health_data associated with the user_id and date
        health_data = fetch_health_data(user_id, date)
        user_info=fetch_user_data(user_id)
        username=user_info['username'].values[0]
        first=user_info['first_name'].values[0]
        last=user_info['last_name'].values[0]
        age = user_info['age'].values[0]
        Respiratory_Rate=health_data['Respiratory_Rate'].values[0]
        spo2 = health_data['Spo2'].values[0]  # Assuming 'spo2' is a column in health_data
        temp=health_data['Temperature'].values[0]
        temp = float(temp)
        heart=health_data['Heart_Rate'].values[0]
        upper_bp=health_data['High_BP'].values[0]
        lower_bp=health_data['Low_BP'].values[0] 
           # Fetch user information from user_info based on user_id and username
       
        gender=user_info['gender'].values[0]
         
        input_data1 = np.array([heart]).reshape(1, -1)  # Example
           # Make predictions using your loaded spo2 model
        prediction1 = model1.predict(input_data1)
           # Process the prediction as needed
        result1 = "Normal" if prediction1 == 0 else "Low Risk" if prediction1 == 1 else "Moderate Risk" if prediction1 == 2 else "High Risk"


        input_data2 = np.array([[upper_bp, lower_bp]]).reshape(1, -1)  # Example
           # Make predictions using your loaded spo2 model
        prediction2 = model2.predict(input_data2)
           # Process the prediction as needed
        result2 = "Normal" if prediction2 == 0 else "Low Risk" if prediction2 == 1 else "Moderate Risk" if prediction2 == 2 else "High Risk"

        input_data3 = np.array([Respiratory_Rate]).reshape(1, -1)  # Example
           # Make predictions using your loaded spo2 model
        prediction3 = model3.predict(input_data3)
           # Process the prediction as needed
        result3 = "Normal" if prediction3 == 0 else "Low Risk" if prediction3 == 1 else "Moderate Risk" if prediction3 == 2 else "High Risk"

        input_data4 = np.array(spo2).reshape(1, -1)  # Example
           # Make predictions using your loaded spo2 model
        prediction4 = model4.predict(input_data4)
           # Process the prediction as needed
        result4 = "Normal" if prediction4 == 0 else "Low Risk" if prediction4 == 1 else "Moderate Risk" if prediction4 == 2 else "High Risk"


        f = (temp * 9/5) + 32
        input_data5 = np.array([temp,f]).reshape(1, -1)  # Example
           # Make predictions using your loaded spo2 model
        prediction5 = model5.predict(input_data5)
           # Process the prediction as needed
        result5 = "Normal" if prediction5 == 0 else "Low Risk" if prediction5 == 1 else "Moderate Risk" if prediction5 == 2 else "High Risk"
        
        score=prediction1+prediction2+prediction3+prediction4+prediction5
        prediction=[]
        if score>=0 and score <= 4:
            prediction ="Low Risk"
        elif score>=5 and score <=6:
            prediction="Moderate Risk"
        else:
            prediction="High Risk"
        # Make predictions using the SPO2 loaded model
        dict={ 'user_id' : user_id,'date':date,'spo2':spo2,'heart':heart,'Respiratory_Rate':Respiratory_Rate,'upper_bp':upper_bp,'lower_bp':lower_bp,
                               'temp':temp,'result1':result1,'result2':result2,'result3':result3,'result4':result4,'result5':result5,
                               'prediction':prediction,'score':score,'username':username,'age':age,'gender':gender,'first':first,'last':last
         }
        
        return render_template("spo2_result.html", user_id=user_id,date=date, 
                               spo2=spo2,heart=heart,Respiratory_Rate=Respiratory_Rate,upper_bp=upper_bp,lower_bp=lower_bp,
                               temp=temp,result1=result1,result2=result2,result3=result3,result4=result4,result5=result5,
                               prediction=prediction,score=score ,data=dict,username=username,first=first,last=last,gender=gender,age=age
                               )
        
    else:
       return render_template("error.html")




@app.route('/get_user_data')
def show_counts():
    # Establish a connection to the SQLite database
    conn = get_db_connection()
    cursor = conn.cursor()
    

    # Query the "users_info" table to count "Y" values in the "heart" column
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE heartproblem ="Y"')
    heart_count = cursor.fetchone()[0]

    # Query the "users_info" table to count "Y" values in the "asthma" column
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE asthma = "Y"')
    asthma_count =cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE highbp = "Y"')
    highbp_count =cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE lowbp = "Y"')
    lowbp_count =cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE diabetes = "Y"')
    diabetes_count =cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE obesity = "Y"')
    obesity_count =cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE none = "Y"')
    none_count =cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE gender = "male"')
    male_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE gender = "female"')
    female_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE age >= 10 AND age < 18;')
    minor_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE age >= 18 AND age < 25;')
    teen_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE age >= 25 AND age < 40;')
    adult_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE age >= 40 AND age < 60;')
    above_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM users_info WHERE age >= 60;')
    old_count = cursor.fetchone()[0]
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT username, first_name,last_name FROM users_info')
    users_data = cursor.fetchall()


    conn.close()
    age_count={
        '10-18year':minor_count,
        '18-25years':teen_count,
        '25-40years':adult_count,
        '40-60years':above_count,
        'above 60 years':old_count

    }
    gen_count = {
        'male': male_count,
        'female': female_count
        }

    # Create a dictionary with the counts
    counts_dict = {
        'heart': heart_count,
        'asthma': asthma_count,
        'highbp':highbp_count,
        'lowbp':lowbp_count,
        'diabetes':diabetes_count,
        'obesity':obesity_count,
        'none':none_count
    }
  # Pass the counts data to the HTML template
    return render_template('piechart.html', counts_dict=counts_dict,gen_count=gen_count,age_count=age_count,users_data=users_data)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
    

           
     