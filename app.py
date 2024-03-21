from flask import Flask, render_template, request, redirect, url_for, session
import mysql.connector
import os

conn = mysql.connector.connect(host="localhost",user="root",password="",database="healthpluse")
cursor = conn.cursor()

app=Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('home.html')
    else:
        return redirect('/')

@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')
    cursor.execute("""SELECT * FROM `users` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(email,password)) 
    users = cursor.fetchall()
    if 'user_id' in session:
        print("yes")
    if len(users) > 0:
        session['user_id'] = users[0][0]
        return redirect('/home')
    else:
        return redirect("/")

@app.route("/add_user",methods=['POST'])
def add_user():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    cursor.execute("""INSERT INTO `users` (`user_id`,`name`,`email`,`password`) VALUES 
                   (NULL,'{}','{}','{}')""".format(name,email,password))
    conn.commit()
    cursor.execute("""SELECT * FROM `users` WHERE `email` LIKE '{}'""".format(email))
    myuser = cursor.fetchall()
    session['user_id'] = myuser[0][0]
    return redirect("/home")
   
if __name__ == "__main__":
    app.run(debug=True)
