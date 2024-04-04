from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import mysql.connector
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import joblib
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
import os
from keras.models import model_from_json
import tensorflow as tf

#<-------------------------- Login System ------------------------>
conn = mysql.connector.connect(host="sql6.freemysqlhosting.net",user="sql6696569",password="hcSUt3jz3w",database="sql6696569 ")
cursor = conn.cursor()

app=Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('mainloading.html')
    else:
        return redirect('/')
    
@app.route('/index')
def index():
    if 'user_id' in session:
        return render_template('home.html')
    else:
        return redirect('/')

@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')
    cursor.execute("""SELECT * FROM `user` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(email,password)) 
    users = cursor.fetchall()
    if len(users) > 0:
        session['user_id'] = users[0][0]
        session['name'] = users[0][1]
        return redirect('/home')
    else:
        return redirect("/")

@app.route("/add_user",methods=['POST'])
def add_user():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    cursor.execute("""INSERT INTO `user` (`user_id`,`name`,`email`,`password`) VALUES 
                   (NULL,'{}','{}','{}')""".format(name,email,password))
    conn.commit()
    cursor.execute("""SELECT * FROM `user` WHERE `email` LIKE '{}'""".format(email))
    myuser = cursor.fetchall()
    session['user_id'] = myuser[0][0]
    return redirect("/home")

@app.route('/logout')
def logout():
    session.pop('user_id')
    return redirect('/')

#<-------------------- Ai Recommendation -------------------->

precautions = pd.read_csv("D:\Projects\HealthPulse  Data Driven strategies for Healthcare Optimization\AI recommendation medicine dataset\precautions_df.csv")
workout = pd.read_csv("D:\Projects\HealthPulse  Data Driven strategies for Healthcare Optimization\AI recommendation medicine dataset\workout_df.csv")
desrciption = pd.read_csv("D:\Projects\HealthPulse  Data Driven strategies for Healthcare Optimization\AI recommendation medicine dataset\description.csv")
medications = pd.read_csv("D:\Projects\HealthPulse  Data Driven strategies for Healthcare Optimization\AI recommendation medicine dataset\medications.csv")
diets = pd.read_csv("D:\Projects\HealthPulse  Data Driven strategies for Healthcare Optimization\AI recommendation medicine dataset\diets.csv")

svc = pickle.load(open("D:\Projects\HealthPulse  Data Driven strategies for Healthcare Optimization\models\svc.pkl",'rb'))

def recommendation_function(dis):
    desc = desrciption[desrciption['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


@app.route('/ai')
def  loadAI():
    if 'user_id' in session:
        return render_template('loadai.html')
    else:
        return redirect('/')

@app.route('/ai_recommendation')
def ai():
    if 'user_id' in session:
        return render_template('Ai.html')
    else:
        return redirect('/')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        sym = request.form.get('sympotms')
        user_sym = [s.strip() for s in sym.split(',')]
        user_sym = [syms.strip("[]' ")for syms in user_sym]
        predicted_disease = get_predicted_value(user_sym)
        desc,pre,med,die,wrkout = recommendation_function(predicted_disease)
    return render_template('Ai.html',predicted_disease = predicted_disease,dis_des=desc,dis_pre=pre[0],dis_med=med,dis_diet=die,dis_workout=wrkout)

#<-------------------------- Heart Prediction ------------------------->

model = joblib.load("models/Heart cardiovascular prediction model.pkl")

@app.route('/loadheart')
def  heart_load():
    if 'user_id' in session:
        return render_template('loadheart.html')
    else:
        return redirect('/')

@app.route('/heart')
def  heart():
    if 'user_id' in session:
        return render_template('heart.html')
    else:
        return redirect('/')

@app.route("/predictheart", methods=["POST"])
def predictheart():
    if request.method == "POST":
        Fn = request.form["fname"]
        age = request.form["ag"]
        mof = "Male"
        getgen = request.form["ge"]
        if getgen[0] == "f" or getgen[0] == "F":
            mof = "Female"
        ctype = "non-anginal"
        chestpain = request.form["cp"]
        if chestpain[1] == "s" or chestpain[1] == "S":
            ctype = "asymptomatic"
        elif chestpain[0] == "a" or chestpain[0] == "A":
            ctype = "atypical angina"
        elif chestpain[0] == "t" or chestpain[0] == "T":
            ctype = "typical angina"

        trb = request.form["trest"]
        ch = request.form["chol"]
        bfb = 1
        fbs = request.form["fb"]
        if fbs[0] == "F" or fbs[0] == "f":
            bfb = 0
        re = 0
        res = request.form["rest"]
        if res[0] == "l" or res[0] == "L":
            re = 1
        elif res[0] == "s" or res[0] == "S":
            re = 2
        thalch = request.form["t"]
        exb = 1
        exhan = request.form["ex"]
        if exhan[0] == "F" or exhan[0] == "f":
            exb = 0
        dep = request.form["de"]
        slo = request.form["sl"]
        sa = 0
        if slo[0] == "u" or slo[0] == "U":
            sa = 1
        elif slo[0] == "d" or slo[0] == "D":
            sa = 2
        c = request.form["ca"]

        b = request.form["bm"]

        inputs = [
            [
                float(age),
                mof,
                ctype,
                float(trb),
                float(ch),
                bfb,
                float(re),
                float(thalch),
                exb,
                float(dep),
                float(sa),
                float(c),
                float(b),
            ]
        ]
        results = model.predict(inputs)
        if str(results[0]) == "0":
            return render_template("noheartdis.html", name=Fn)
        else:
            return render_template("heartresult.html", name=Fn, stage=str(results[0]))
    return render_template("heart.html")


#<------------------ Lungs cancer  ------------------>

@app.route('/loadlungs')
def loadinglungs():
    if 'user_id' in session:
        return render_template('loadinglung.html')
    else:
        return redirect('/')

@app.route('/mainlungs')
def mainlungs():
    if 'user_id' in session:
        return render_template('lungmain.html')
    else:
        return redirect('/')


@app.route('/lungs')
def lungs():
    if 'user_id' in session:
        return render_template('lung.html')
    else:
        return redirect('/')

@app.route("/predictlungs", methods=["POST"])
def predictlungs():
    if request.method == "POST":
        name=request.form['fname']
        gender = request.form['Gender']
        if gender[0] == 'M':
            genderin=1
        else:
            genderin = 0
        age = int(request.form['ag'])
        Smoking = request.form['Smoking']
        if Smoking ==  'Yes':
            Smokingin = 1
        else:
            Smokingin = 0
        yellow = request.form['Yellow']
        if yellow == 'Yes':
            yellowness = 1
        else:
            yellowness = 0
        anxiety = request.form['Anxiety']
        if  anxiety == 'Yes':
            anxietylevel = 1
        else :  
            anxietylevel = 0
        peer = request.form['Peer']
        if peer=='Yes':
            peers= 1
        else:
            peers =0
        Chronic = request.form['Chronic']
        if  Chronic=="Yes":
           Chro=1
        else:
            Chro=0
        Fatigue = request.form['Fatigue']
        if Fatigue  == 'Yes':
            Fatiguein=1
        else:
            Fatiguein = 0 
        Allergy = request.form['Allergy']
        if  Allergy  == 'Yes':
            Allergyin = 1
        else:
            Allergyin = 0
        Wheezing = request.form['Wheezing']
        if Wheezing == 'Yes':
            Wheezingin = 1
        else:
            Wheezingin = 0
        Alcohol = request.form['Alcohol']
        if Alcohol ==  'Yes':
            Alcoholin = 1
        else:
            Alcoholin = 0
        Coughing = request.form['Coughing']
        if  Coughing == "Yes":
            Coughingin = 1
        else:
            Coughingin = 0
        Shortness = request.form['Shortness']
        if Shortness ==  "Yes" :
            Shortnessin = 1
        else:
            Shortnessin =0
        Swallowing = request.form['Swallowing']
        if  Swallowing == "Yes":
            Swallowingin = 1
        else:
            Swallowingin = 0
        Chest = request.form['Chest']
        if Chest ==  "Yes" :
            Chestin = 1
        else:
            Chestin = 0
    lunginput=[[genderin,age,Smokingin,yellowness,anxietylevel,peers,Chro,Fatiguein,Allergyin,Wheezingin,Alcoholin,Coughingin,Shortnessin,Swallowingin,Chestin]]
    lungmodel = joblib.load("models/lung_cancer_predictor_model.pkl")
    lungmodelresult = lungmodel.predict(lunginput)
    
    if str(lungmodelresult[0]) == '1':
        return render_template('lungsresult.html',name=name)
    else:
        return render_template('nolungsdis.html',name=name)

#<------------------------   Kidney ----------------------->
@app.route('/loadkidney')
def loadkidney():
    if 'user_id' in session:
        return render_template('loadkidney.html')
    else:
        return redirect('/')

@app.route('/mainkidney')
def mainkidney():
    if 'user_id' in session:
        return render_template('kidneymain.html')
    else:
        return redirect('/')

#<---------------- Brain Tumor ------------------------------>

class_names = ["pituitary_tumor", "no_tumor", "meningioma_tumor", "glioma_tumor"]
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}


UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
json_file = open("model_tumor.json", "r")
loaded_model_json = json_file.read()
json_file.close()
cnn_model = model_from_json(loaded_model_json)
cnn_model.load_weights("models/model_tumor.h5")

IMAGE_SIZE = 150


# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range

    return image

# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)

    return preprocess_image(image)

# Predict & classify image
def classify(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    prob = cnn_model.predict(preprocessed_imgage)
    # Vector of probabilities
    pred_labels = np.argmax(prob, axis=1)
    label = class_names[pred_labels[0]]
    classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]

    return label, classified_prob

@app.route('/loadbrain')
def loadbrain():
    if 'user_id' in session:
        return render_template('loadbrain.html')
    else:
        return redirect('/')
    
@app.route('/mainbrain')
def mainbrain():
    if 'user_id' in session:
        return render_template('brainmain.html')
    else:
        return redirect('/')

@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if 'user_id' in session:
        if request.method == "GET":
            return render_template("brainmain.html")

        else:
            file = request.files["image"]
            upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            print(upload_image_path)
            file.save(upload_image_path)
            label, prob = classify(cnn_model, upload_image_path)
            prob = round((prob * 100), 2)
        return render_template("classify.html", image_file_name=file.filename, label=label, result=prob)
    else:
        return redirect('/')
    
@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)