"""
ULTIMATE MYOPIA DETECTION APPLICATION
Hybrid version combining the best features from all previous versions

Features:
- G.py: Data management, schema migration, image storage, multi-class predictions
- improved_app.py: Password hashing, session management, input validation
- Main_Code.py: Model caching, image preprocessing

This is production-ready!
"""

import streamlit as st
import sqlite3
from PIL import Image
import io
import re
import numpy as np
import tensorflow as tf
from datetime import datetime
import hashlib
import os
import time  # ← Added for sleep function

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_NAME = "evaluation-8.db"
MODEL_PATH = "myopia_model_densenet_F.h5"
IMAGE_SIZE = (224, 224)
MAX_IMAGE_SIZE_MB = 10
MIN_IMAGE_DIMENSION = 100

# Class labels (3-class classification)
CLASS_LABELS = ["Normal", "High Myopia", "Pathological Myopia"]

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Myopia Detection", layout="wide")

# ============================================================================
# DATABASE AUTO-CREATION NOTICE
# ============================================================================
def show_db_status():
    """Show database status on first run."""
    if not os.path.exists(DB_NAME):
        st.warning(f"⚠️ Database '{DB_NAME}' will be created automatically on first initialization...")
        st.info(f"📍 Database location: {os.path.abspath(DB_NAME)}")
    return os.path.exists(DB_NAME)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def hash_password(password):
    """Hash password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def require_login():
    """Check if user is logged in."""
    if not st.session_state.logged_in:
        st.error("❌ Please login first!")
        st.stop()

def navigate_to(page_name):
    """Navigate to a page."""
    st.session_state.page = page_name
    st.rerun()

def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Validate phone number."""
    return phone.isdigit() and len(phone) == 10

def validate_fullname(name):
    """Validate full name."""
    return re.match(r'^[A-Za-z ]+$', name.strip()) is not None

def validate_password(password):
    """Validate password strength."""
    special_sym = {'$', '@', '#', '%'}
    return (8 <= len(password) <= 20 and
            any(char.isdigit() for char in password) and
            any(char.isupper() for char in password) and
            any(char.islower() for char in password) and
            any(char in special_sym for char in password))

def validate_and_prepare_image(uploaded_file):
    """Validate and prepare image."""
    try:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_IMAGE_SIZE_MB:
            st.error(f"❌ Image too large (max {MAX_IMAGE_SIZE_MB} MB)")
            return None
        
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        
        width, height = image.size
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            st.error(f"❌ Image too small (min {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION})")
            return None
        
        return image.resize(IMAGE_SIZE)
    except Exception as e:
        st.error(f"❌ Invalid image: {str(e)}")
        return None

# ============================================================================
# DATABASE INITIALIZATION (From G.py with enhancements)
# ============================================================================

@st.cache_resource
def init_db():
    """
    Initialize database with schema migration support.
    
    AUTO-CREATES if it doesn't exist:
    - Database file (evaluation-8.db)
    - All required tables
    - All required columns
    """
    try:
        # Check if database needs to be created
        db_exists = os.path.exists(DB_NAME)
        
        with sqlite3.connect(DB_NAME) as db:
            cursor = db.cursor()
            
            # Create admin_registration table
            cursor.execute("""CREATE TABLE IF NOT EXISTS admin_registration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                phoneno TEXT NOT NULL,
                gender TEXT NOT NULL,
                age INTEGER NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
            
            # Create patient_info_1 table
            cursor.execute("""CREATE TABLE IF NOT EXISTS patient_info_1 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                contact TEXT NOT NULL,
                address TEXT NOT NULL,
                image BLOB,
                registration_date TEXT,
                registration_time TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
            
            # Create patient_predictions table
            cursor.execute("""CREATE TABLE IF NOT EXISTS patient_predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL,
                prediction_date TEXT,
                prediction_time TEXT,
                prediction_image BLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(patient_id) REFERENCES patient_info_1(id)
            )""")
            
            # Schema migration function (from G.py)
            def add_column_if_not_exists(table_name, column_name, column_type):
                """Safely add columns to existing tables."""
                try:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    existing_columns = [column[1] for column in cursor.fetchall()]
                    
                    if column_name not in existing_columns:
                        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                except sqlite3.OperationalError:
                    pass
            
            # Add missing columns for backward compatibility
            add_column_if_not_exists('patient_info_1', 'registration_date', 'TEXT')
            add_column_if_not_exists('patient_info_1', 'registration_time', 'TEXT')
            add_column_if_not_exists('patient_predictions', 'confidence', 'REAL')
            add_column_if_not_exists('patient_predictions', 'prediction_date', 'TEXT')
            add_column_if_not_exists('patient_predictions', 'prediction_time', 'TEXT')
            add_column_if_not_exists('patient_predictions', 'prediction_image', 'BLOB')
            
            db.commit()
        
        # Show status message on first run
        if not db_exists:
            st.success(f"✅ Database '{DB_NAME}' created successfully!")
            st.info(f"📍 Location: {os.path.abspath(DB_NAME)}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Database initialization failed: {str(e)}")
        st.error(f"Make sure you have write permissions in: {os.path.abspath('.')}")
        return False

# Initialize database (auto-creates if needed)
db_initialized = init_db()

if not db_initialized:
    st.error("❌ Failed to initialize database. Please check permissions and try again.")
    st.stop()

# ============================================================================
# MODEL LOADING (Cached - from Main_Code.py)
# ============================================================================

@st.cache_resource
def load_model():
    """Load and cache TensorFlow model."""
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_myopia(image, model):
    """Make myopia prediction with multi-class support (from G.py)."""
    try:
        img_array = np.array(image).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        result = CLASS_LABELS[predicted_class_idx]
        
        return predictions, predicted_class_idx, result, confidence
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None, None, None

# ============================================================================
# PAGE COMPONENTS
# ============================================================================

def home_page():
    """Home page."""
    st.markdown(
        "<h1 style='text-align: center; color: white; background-color: #0066cc; padding: 20px; border-radius: 10px;'>"
        "👁️ Automatic Detection of Myopia</h1>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("### Welcome to Advanced Myopia Detection System")
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            if st.button("🔐 Login", use_container_width=True):
                navigate_to("login")
        
        with col_right:
            if st.button("✍️ Register", use_container_width=True):
                navigate_to("register")

def login_page():
    """Login page (with password hashing from improved_app.py)."""
    st.markdown(
        "<h2 style='text-align: center; color: white; background-color: #28a745; padding: 15px;'>"
        "🔐 SIGN-IN</h2>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")
        
        if st.button("🔓 Login", use_container_width=True):
            if not username or not password:
                st.error("❌ Please enter username and password")
            else:
                with sqlite3.connect(DB_NAME) as db:
                    cursor = db.cursor()
                    password_hash = hash_password(password)
                    cursor.execute(
                        "SELECT id, username FROM admin_registration WHERE username = ? AND password_hash = ?",
                        (username, password_hash)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        st.session_state.logged_in = True
                        st.session_state.current_user = result[1]
                        st.success("✅ Login successful!")
                        st.balloons()
                        time.sleep(1)
                        navigate_to("main")
                    else:
                        st.error("❌ Invalid credentials")
        
        st.markdown("---")
        if st.button("📝 Create Account", use_container_width=True):
            navigate_to("register")

def registration_page():
    """Registration page (with password hashing)."""
    st.markdown(
        "<h2 style='text-align: center; color: white; background-color: #28a745; padding: 15px;'>"
        "✍️ SIGN-UP</h2>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        fullname = st.text_input("👤 Full Name")
        email = st.text_input("📧 Email")
        phoneno = st.text_input("📱 Phone Number")
        gender = st.radio("⚤ Gender", ("Male", "Female"), horizontal=True)
        age = st.number_input("🎂 Age", min_value=1, max_value=120)
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")
        password_confirm = st.text_input("🔑 Confirm Password", type="password")
        
        if st.button("✅ Submit", use_container_width=True):
            errors = []
            
            if not fullname.strip():
                errors.append("Full name is required")
            elif not validate_fullname(fullname):
                errors.append("Invalid full name (letters and spaces only)")
            
            if not validate_email(email):
                errors.append("Invalid email format")
            
            if not validate_phone(phoneno):
                errors.append("Phone number must be 10 digits")
            
            if not validate_password(password):
                errors.append("Weak password (8-20 chars, 1 digit, 1 uppercase, 1 lowercase, 1 special)")
            
            if password != password_confirm:
                errors.append("Passwords do not match")
            
            if not username.strip():
                errors.append("Username is required")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                with sqlite3.connect(DB_NAME) as db:
                    cursor = db.cursor()
                    cursor.execute("SELECT id FROM admin_registration WHERE username = ?", (username,))
                    if cursor.fetchone():
                        st.error("❌ Username already exists")
                    else:
                        try:
                            password_hash = hash_password(password)
                            cursor.execute(
                                "INSERT INTO admin_registration (fullname, username, email, phoneno, gender, age, password_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (fullname, username, email, phoneno, gender, age, password_hash)
                            )
                            db.commit()
                            st.success("✅ Account created! Redirecting to login...")
                            time.sleep(2)
                            navigate_to("login")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

def main_page():
    """Main menu."""
    require_login()
    
    st.markdown(f"<h2 style='color: #0066cc;'>👋 Welcome, {st.session_state.current_user}!</h2>", unsafe_allow_html=True)
    
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("---")
    
    if st.sidebar.button("➕ New Patient", use_container_width=True):
        navigate_to("new_patient")
    if st.sidebar.button("🔬 Predict Myopia", use_container_width=True):
        navigate_to("predict")
    if st.sidebar.button("📊 Patient History", use_container_width=True):
        navigate_to("patient_history")
    
    st.sidebar.markdown("---")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("🏠 Home", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            navigate_to("home")
    
    with col2:
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.success("✅ Logged out!")
            time.sleep(1)
            navigate_to("home")

def new_patient_page():
    """Register new patient (from G.py with improvements)."""
    require_login()
    
    st.title("➕ Patient Registration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        name = st.text_input("👤 Patient Name")
        age = st.number_input("🎂 Age", min_value=0, max_value=120)
        gender = st.radio("⚤ Gender", ("Male", "Female"), horizontal=True)
        contact = st.text_input("📱 Contact Number")
        address = st.text_input("📍 Address")
        uploaded_image = st.file_uploader("📷 Upload Eye Image", type=["png", "jpg", "jpeg"])
        
        if uploaded_image:
            image_preview = validate_and_prepare_image(uploaded_image)
            if image_preview:
                st.image(image_preview, caption="Image Preview")
        
        col_submit, col_back = st.columns(2)
        
        with col_submit:
            if st.button("✅ Register Patient", use_container_width=True):
                # Comprehensive validation
                errors = []
                
                # Validate name
                if not name or not name.strip():
                    errors.append("Patient name is required")
                elif not validate_fullname(name):
                    errors.append("❌ Patient name must contain ONLY letters and spaces")
                    st.error("Examples:\n✅ 'Rajesh Kumar' (CORRECT)\n❌ 'Rajesh123' (has numbers)\n❌ '9874563210' (only numbers)")
                
                # Validate contact
                if not contact or not contact.strip():
                    errors.append("Contact number is required")
                elif not validate_phone(contact):
                    errors.append(f"❌ Contact must be EXACTLY 10 digits. You entered {len(contact)} digits")
                    st.error("Examples:\n✅ '9876543210' (CORRECT - 10 digits)\n❌ '87564654985' (11 digits)\n❌ '123456789' (9 digits)")
                
                # Validate address
                if not address or not address.strip():
                    errors.append("Address is required")
                
                # Validate image
                if not uploaded_image:
                    errors.append("Eye image is required")
                
                # Display all errors
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    image_data = validate_and_prepare_image(uploaded_image)
                    if image_data:
                        try:
                            img_byte_arr = io.BytesIO()
                            image_data.save(img_byte_arr, format='JPEG')
                            image_bytes = img_byte_arr.getvalue()
                            
                            now = datetime.now()
                            reg_date = now.strftime("%Y-%m-%d")
                            reg_time = now.strftime("%H:%M:%S")
                            
                            with sqlite3.connect(DB_NAME) as db:
                                cursor = db.cursor()
                                cursor.execute(
                                    "INSERT INTO patient_info_1 (name, age, gender, contact, address, image, registration_date, registration_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                    (name, age, gender, contact, address, image_bytes, reg_date, reg_time)
                                )
                                db.commit()
                                patient_id = cursor.lastrowid
                            
                            st.success(f"✅ Patient registered! ID: {patient_id}")
                            st.info(f"📅 {reg_date} at {reg_time}")
                            st.balloons()
                            time.sleep(2)
                            navigate_to("main")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with col_back:
            if st.button("🔙 Back", use_container_width=True):
                navigate_to("main")

def prediction_page():
    """Make prediction (from G.py with multi-class support)."""
    require_login()
    
    st.title("🔬 Myopia Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        patient_id = st.number_input("👤 Enter Patient ID", min_value=1, step=1)
        
        # Show patient info (from G.py)
        if patient_id:
            with sqlite3.connect(DB_NAME) as db:
                cursor = db.cursor()
                cursor.execute("SELECT name, age, gender FROM patient_info_1 WHERE id = ?", (patient_id,))
                patient_info = cursor.fetchone()
                if patient_info:
                    st.info(f"👤 {patient_info[0]} | Age: {patient_info[1]} | {patient_info[2]}")
                else:
                    st.warning("⚠️ Patient ID not found")
        
        uploaded_image = st.file_uploader("📷 Upload Eye Image", type=["png", "jpg", "jpeg"])
        
        if uploaded_image:
            image_data = validate_and_prepare_image(uploaded_image)
            if image_data:
                st.image(image_data, caption="Image Preview")
        
        col_predict, col_back = st.columns(2)
        
        with col_predict:
            if st.button("🔍 Predict", use_container_width=True):
                if not patient_id:
                    st.error("❌ Enter Patient ID")
                elif not uploaded_image:
                    st.error("❌ Upload image")
                else:
                    image_data = validate_and_prepare_image(uploaded_image)
                    if image_data:
                        with st.spinner("🔄 Analyzing..."):
                            predictions, predicted_class_idx, result, confidence = predict_myopia(image_data, model)
                            
                            if predictions is not None:
                                # Display results
                                st.success(f"✅ {result}")
                                
                                col_r1, col_r2 = st.columns(2)
                                with col_r1:
                                    st.metric("Prediction", result)
                                with col_r2:
                                    st.metric("Confidence", f"{confidence*100:.1f}%")
                                
                                # Show all probabilities (from G.py)
                                st.markdown("**📊 All Class Probabilities:**")
                                for label, prob in zip(CLASS_LABELS, predictions):
                                    st.write(f"- {label}: {prob:.2%}")
                                
                                # Save prediction
                                try:
                                    uploaded_image.seek(0)
                                    image_bytes = uploaded_image.read()
                                    
                                    now = datetime.now()
                                    pred_date = now.strftime("%Y-%m-%d")
                                    pred_time = now.strftime("%H:%M:%S")
                                    
                                    with sqlite3.connect(DB_NAME) as db:
                                        cursor = db.cursor()
                                        cursor.execute(
                                            "INSERT INTO patient_predictions (patient_id, prediction, confidence, prediction_date, prediction_time, prediction_image) VALUES (?, ?, ?, ?, ?, ?)",
                                            (patient_id, result, confidence, pred_date, pred_time, image_bytes)
                                        )
                                        db.commit()
                                    
                                    st.success("✅ Prediction saved!")
                                except Exception as e:
                                    st.error(f"❌ Save error: {str(e)}")
        
        with col_back:
            if st.button("🔙 Back", use_container_width=True):
                navigate_to("main")

def patient_history_page():
    """View patient history (from G.py with expandable cards)."""
    require_login()
    
    st.title("📊 Patient History")
    
    patient_id = st.number_input("Enter Patient ID", min_value=1, step=1)
    
    if st.button("🔍 Fetch Patient Info"):
        with sqlite3.connect(DB_NAME) as db:
            cursor = db.cursor()
            
            # Fetch patient info
            cursor.execute("SELECT * FROM patient_info_1 WHERE id = ?", (patient_id,))
            record = cursor.fetchone()
            
            if record:
                st.markdown("### Patient Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Name", record[1])
                    st.metric("Age", record[2])
                    st.metric("Gender", record[3])
                
                with col2:
                    st.metric("Contact", record[4])
                    st.metric("Address", record[5])
                    if record[7]:
                        st.caption(f"📅 Registered: {record[7]} {record[8] if record[8] else ''}")
                
                if record[6]:
                    st.markdown("### Patient Image")
                    try:
                        image = Image.open(io.BytesIO(record[6]))
                        st.image(image, width=200)
                    except:
                        st.error("Could not load image")
                
                # Prediction history
                st.markdown("### Prediction History")
                
                try:
                    cursor.execute("PRAGMA table_info(patient_predictions)")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    query_columns = ["prediction"]
                    if "confidence" in columns:
                        query_columns.append("confidence")
                    if "prediction_date" in columns:
                        query_columns.append("prediction_date")
                    if "prediction_time" in columns:
                        query_columns.append("prediction_time")
                    if "prediction_image" in columns:
                        query_columns.append("prediction_image")
                    
                    query = f"SELECT {', '.join(query_columns)} FROM patient_predictions WHERE patient_id = ? ORDER BY prediction_id DESC"
                    cursor.execute(query, (patient_id,))
                    predictions = cursor.fetchall()
                    
                    if predictions:
                        for i, pred in enumerate(predictions, 1):
                            with st.expander(f"Prediction #{i} - {pred[0]}", expanded=(i==1)):
                                col_p1, col_p2 = st.columns(2)
                                
                                with col_p1:
                                    st.write(f"**🎯 Result:** {pred[0]}")
                                    
                                    if len(pred) > 1 and "confidence" in query_columns:
                                        conf_idx = query_columns.index("confidence")
                                        if pred[conf_idx]:
                                            st.write(f"**📊 Confidence:** {pred[conf_idx]:.1%}")
                                    
                                    if "prediction_date" in query_columns:
                                        date_idx = query_columns.index("prediction_date")
                                        if len(pred) > date_idx and pred[date_idx]:
                                            st.write(f"**📅 {pred[date_idx]}**")
                                
                                with col_p2:
                                    if "prediction_image" in query_columns:
                                        img_idx = query_columns.index("prediction_image")
                                        if len(pred) > img_idx and pred[img_idx]:
                                            try:
                                                pred_img = Image.open(io.BytesIO(pred[img_idx]))
                                                st.image(pred_img, width=150)
                                            except:
                                                st.write("No image")
                    else:
                        st.info("No predictions yet")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Patient not found")
    
    if st.button("🔙 Back"):
        navigate_to("main")

# ============================================================================
# MAIN APP ROUTING
# ============================================================================

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "register":
    registration_page()
elif st.session_state.page == "main":
    main_page()
elif st.session_state.page == "new_patient":
    new_patient_page()
elif st.session_state.page == "predict":
    prediction_page()
elif st.session_state.page == "patient_history":
    patient_history_page()