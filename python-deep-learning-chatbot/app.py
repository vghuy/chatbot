from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_login import LoginManager, login_user, UserMixin, login_required, logout_user
import hashlib
import os
import pyodbc
import processor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'

login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

def get_user_by_username(username):
    server = 'DESKTOP-56CFC87'
    database = 'Chatbot'
    db_username = 'sa'
    db_password = '123'
    driver = '{SQL Server}'
    conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={db_username};PWD={db_password}'

    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    query = "SELECT * FROM Users WHERE username = ?"
    cursor.execute(query, (username,))
    user = cursor.fetchone()

    conn.close()
    return user

@app.route('/', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        
        user = get_user_by_username(username)

        if user:
            db_stored_password = user[2]  # Chỉ số cột chứa mật khẩu trong tuple 'user'

            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            if hashed_password == db_stored_password:
                user_id = user[0]  # Chỉ số cột chứa ID người dùng trong tuple 'user'
                login_user(User(user_id))
                return redirect(url_for('index'))  # Chuyển hướng đến route '/index' sau khi đăng nhập thành công

        return render_template('login.html', error_message="Tài khoản hoặc mật khẩu không đúng")

    return render_template('login.html')

@app.route('/index', methods=["GET"])
@login_required
def index():
    return render_template('index.html')

@app.route('/logout', methods = ["POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/chatbot', methods=["POST"])
def chatbot_response():
    response = "No response"  # Gán một giá trị mặc định

    if request.method == 'POST':
        the_question = request.form['question']
        response = processor.chatbot_response(the_question)

    return jsonify({"response": response})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        server = 'DESKTOP-56CFC87'
        database = 'Chatbot'
        db_username = 'sa'
        db_password = '123'
        driver = '{SQL Server}'
        conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={db_username};PWD={db_password}'

        # Mã hóa mật khẩu và thực hiện chèn thông tin người dùng vào cơ sở dữ liệu
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()

        return redirect(url_for('login'))

    return render_template('register.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
