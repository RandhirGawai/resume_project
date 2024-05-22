from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from werkzeug.security import generate_password_hash, check_password_hash  # Add this line
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask import Flask, render_template, request, redirect, url_for, flash,jsonify
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import csv
from docx import Document
import textract
from PyPDF2 import PdfReader
import os
import csv
from wtforms import BooleanField

# ...
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your_secret_key'
login_manager = LoginManager(app)
login_manager.login_view = 'login'

USERS_CSV = 'data/users.csv'

class User(UserMixin):
    def __init__(self, id, username, email, password):
        self.id = id
        self.username = username
        self.email = email
        self.password = password

    def get_id(self):
        return self.id

def load_users():
    users = []
    with open(USERS_CSV, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            user = User(row['id'], row['username'].lower(), row['email'], row['password'])
            users.append(user)
    print("Loaded Users:", users)
    return users





users = load_users()

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

# Save user to CSV
# Save the user to CSV
# Remove this line from save_user function
# hashed_password = generate_password_hash(user.password, method='pbkdf2:sha256')

# Save the user to CSV
def save_user(user):
    with open(USERS_CSV, mode='a', newline='') as csvfile:
        fieldnames = ['id', 'username', 'email', 'password']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'id': user.id, 'username': user.username, 'email': user.email, 'password': user.password})
    
    # Append the new user to the users list
    users.append(user)

def check_user_credentials(username, password):
    user = next((user for user in users if user.username == username.lower()), None)
    if user and check_password_hash(user.password, password):
        return True
    return False

@login_manager.user_loader
def load_user(user_id):
    return next((user for user in users if user.id == user_id), None)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        # Check if the username or email is already registered
        if any(user.username == form.username.data or user.email == form.email.data for user in users):
            flash('Username or email is already registered. Choose a different one.', 'danger')
        else:
            # Hash the password before storing it
            hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')

            # Create a new user
            new_user = User(id=str(len(users) + 1), username=form.username.data, email=form.email.data, password=hashed_password)
            
            # Save the user to CSV and users list
            save_user(new_user)

            flash('Registration successful', 'success')
            return redirect(url_for('login'))  # Redirect to the login page

    return render_template('register.html', form=form)

# Load the pre-processed data from the pickle file
new_ds = pickle.load(open('preprocess_data.pkl', 'rb'))

# Preprocess the skills to create a TF-IDF representation
tfidf_vectorizer = TfidfVectorizer()
skills_matrix = tfidf_vectorizer.fit_transform(new_ds['Post'])

# Function to extract text from different file formats
def extract_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        return extract_text_from_text_file(file_path)

from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    pdf = PdfReader(open(pdf_path, 'rb'))  # Use PdfReader instead of PdfFileReader

    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return text

def extract_text_from_text_file(text_file_path):
    text = textract.process(text_file_path).decode('utf-8')
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text

# Function to get the top words in a text
def get_top_words(text, n=50):
    words = text.split()
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:n]
    return [word[0] for word in top_words]

# Function to analyze a resume
def analyze_resume(file_path, new_ds):
    resume_text = extract_text_from_file(file_path)
    cleaned_resume_text = preprocess_text(resume_text)
    all_company_descriptions = new_ds['Combo'].tolist()
    top_dataset_words = get_top_words(' '.join(all_company_descriptions))
    top_resume_words = get_top_words(cleaned_resume_text)
    top_words = list(set(top_resume_words + top_dataset_words))
    top_words_file_path = 'top_resume_words.csv'
    #save_top_words_to_csv(os.path.basename(file_path), top_words, top_words_file_path)
    tfidf_vectorizer = TfidfVectorizer(vocabulary=top_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform([cleaned_resume_text] + all_company_descriptions)
    cosine_similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()
    new_ds['Similarity Score'] = (cosine_similarities * 100).astype(int)
    threshold = 0.1
    filtered_df = new_ds[new_ds['Similarity Score'] >= threshold]
    filtered_df = filtered_df.sort_values(by='Similarity Score', ascending=False)
    top_5_companies = filtered_df.nlargest(5, 'Similarity Score')
    top_5_companies['Resume Rating'] = top_5_companies['Similarity Score']+40
    top_companies = top_5_companies[['Company', 'Post', 'Experience', 'Location', 'JD','Openings','Applicants','Skill', 'Resume Rating']].to_dict(orient='records')
    return top_companies

# Function to suggest companies based on a skill
def suggest_companies(skill, new_ds):
    skill_vector = tfidf_vectorizer.transform([skill])
    cosine_similarities = cosine_similarity(skill_vector, skills_matrix)
    top_indices = cosine_similarities.argsort()[:, -5:][0]
    suggestions = new_ds.iloc[top_indices][['Company', 'Post', 'Experience', 'Location','Openings','Applicants','Skill','JD']].to_dict(orient='records')
    return suggestions

from werkzeug.security import generate_password_hash

# ... (other imports)
from werkzeug.security import check_password_hash  # Add this line

# ... (other imports)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        if check_user_credentials(form.username.data, form.password.data):
            user = next((user for user in users if user.username == form.username.data.lower()), None)
            if user:
                login_user(user, remember=form.remember.data)
                flash('Login successful', 'success')
                print(f"User '{user.username}' logged in successfully.")
                return redirect(url_for('index'))
            else:
                flash('Login failed. Check your username and password.', 'danger')
                print(f"Login failed for username: {form.username.data}")
        else:
            flash('Login failed. Check your username and password.', 'danger')
            print(f"Login failed for username: {form.username.data}")

    return render_template('login.html', form=form)





@app.route('/')
@login_required
def index():
    return render_template('index.html')



@app.route('/<path:invalid_path>')
def handle_invalid_path(invalid_path):
    flash(f'Invalid path: {invalid_path}', 'danger')
    return redirect(url_for('index'))





# Route for resume analysis
@app.route('/RESUME_ANALYZER', methods=['GET', 'POST'])
@login_required

def analyze_resume_route():
    print("got the function")
    return render_template("RESUME_ANALYZER.html")

@app.route('/topics_detail')
@login_required

def topics_detail():
    # Your view logic here
    pass

@app.route('/resume_analyzer')
@login_required

def resume_analyzer():
    return render_template('resume_analyzer.html')

@app.route("/resume_analyses_result", methods=['GET', 'POST'])
@login_required
def analyze():
    if request.method == "POST":
        if 'resume' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            # Ensure the uploads directory exists
            upload_folder = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            analysis_result = analyze_resume(file_path, new_ds)
            # Pass the template name and the analysis_result as separate variables
            return render_template('analysis_result.html', analysis_result=analysis_result)

    return render_template('resume_analyzer.html')



# Route for skill-based company suggestion
@app.route('/SUGGESTION', methods=['GET', 'POST'])
@login_required

def suggest_skill():
    skill = request.form.get('field')
    if not skill:
        return jsonify({'error': 'Skill is required'})

    suggested_companies = suggest_companies(skill, new_ds)
    return render_template('SUGGESTION_result.html', suggested_companies=suggested_companies)
# Function to save a new blog to the CSV file

def save_blog(title, content):
    csv_file_path = os.path.join("data", "blogs.csv")
    
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(csv_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, title, content])


# Function to read blogs from the CSV file
# Function to read blogs from the CSV file
# Function to read blogs from the CSV file
def read_blogs():
    blogs = []
    with open("data/blogs.csv", mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            blogs.append({"timestamp": row[0], "title": row[1], "content": row[2]})
    return blogs


@app.route("/blogs", methods=['GET'])
@login_required

def get_blogs():
    blogs = read_blogs()
    return jsonify(blogs)
@app.route("/blog", methods=['GET', 'POST'])
@login_required

def add_blog():
    if request.method == 'POST':
        title = request.form.get("blogTitle")
        content = request.form.get("blogContent")

        if title and content:
            save_blog(title, content)

    
    # After saving the blog, redirect to the blog page
    return render_template("blog.html", blogs=read_blogs())

from flask_paginate import Pagination
import csv
PER_PAGE = 10  # Number of items per page
csv_folder = 'all_job_files/'

def load_csv_data(dataset):
    csv_data = []  # Store the CSV data
    csv_file_path = os.path.join(csv_folder, f'{dataset}.csv')

    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            csv_data.append(row)

    return csv_data
def get_page_parameter():
    page = request.args.get('page', type=int, default=1)
    return page

# Route for Data Science
@app.route('/data_science')
@login_required

def show_data_science():
    return show_dataset('DataScience')

# Route for Salesforce
@app.route('/salesforce')
@login_required

def show_salesforce():
    return show_dataset('Salesforce')

# Route for Fullstack Web Development
@app.route('/fullstack_web')
@login_required

def show_fullstack_web():
    return show_dataset('FullstackWebDevelopment')

# Route for Mobile App Development
@app.route('/sql_database')
@login_required

def show_sql_database():
    return show_dataset('sqldatabase')

from flask_login import current_user, login_required

@app.route('/your_route')
@login_required
def your_route():
    return render_template('your_template.html', user_authenticated=True, current_user=current_user)

# Common function to display dataset
def show_dataset(dataset):
    # Make sure the dataset name is valid for a file name
    valid_dataset_name = re.sub(r'\W+', '', dataset)

    # Construct the correct file path
    csv_file_path = os.path.join(csv_folder, f'{valid_dataset_name}.csv')

    # Check if the file exists
    if not os.path.exists(csv_file_path):
        return render_template('error.html', error_message=f'CSV file not found for dataset: {dataset}')

    # Get the page number from the URL parameters
    page = get_page_parameter()

    # Load all rows from the CSV
    data = load_csv_data(valid_dataset_name)

    # Calculate the total number of items
    total_items = len(data)

    # Create a Pagination object
    pagination = Pagination(page=page, total=total_items, per_page=PER_PAGE, css_framework='bootstrap4')

    # Calculate the range for the current page
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE

    # Get the data for the current page
    pagination_data = data[start:end]

    return render_template('index1.html', dataset=valid_dataset_name, data=pagination_data, pagination=pagination)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        print(f"An error occurred: {e}")

