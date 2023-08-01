# -*- coding: utf-8 -*-
"""
Created on Sat May 20 20:59:33 2023

@author: lEO
"""

# CLASSIFICATION MACHINE LEARNING FULL CODE
# I M P O R T   L I B R A R I E S 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz
from ydata_profiling import ProfileReport
import warnings
import streamlit as st
import streamlit_option_menu as som
from st_on_hover_tabs import on_hover_tabs
# from st_aggrid import 
import base64

# Data transformation
from sklearn.preprocessing import LabelEncoder
# Fixing unbalanced dataset
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# Feature selection
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_classif
# Splitting the data
from sklearn.model_selection import train_test_split
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from lazypredict.Supervised import LazyClassifier
from xgboost import XGBClassifier
# Selecting the best K for KNN
from sklearn.model_selection import GridSearchCV
# Cross validation
from sklearn.model_selection import cross_val_score
# Model evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve


# D E A L   W I T H   W A R N I N G S 
warnings.filterwarnings('ignore')

# Streamlit app
st.set_page_config(
    page_title = "Titanic Survival Predictor",
    page_icon = "🌊",
    layout = "wide"
    )


# Streamlit functions
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-repeat: repeat;
        background-attachment: scroll;
    }}
    </style>
    """,
        unsafe_allow_html = True
    )
    
background = add_bg_from_local("C:/Users/lEO/Downloads/magicpattern-grid-pattern-1689780611478.png")

# STEP 1: Store pages inside session_state
if 'page' not in st.session_state:
    st.session_state['page'] = 0

def next_page():
    st.session_state['page'] += 1

def previous_page():
    st.session_state['page'] -= 1


# index = [n for n in range(1, 20)]
dataset = pd.read_csv("C:/Users/lEO/Desktop/GMC Dataset/Data Exploration with Pandas/titanic-passengers.csv", delimiter = ';')

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
   
with st.sidebar:
    tabs = on_hover_tabs(tabName = ["Home", "Dataset", "EDA", "App", "Evaluation", "Graphs", "About us", "Contact"], 
                         iconName= ['home', 'publish', 'hub', 'api', 'dashboard', 'equalizer', 'handshake', 'person'], default_choice=0)


if tabs == "Home":
    st.title("Welcome aboard to Titanic Explorer! 👋")
    st.divider()
    st.markdown("""
    Ahoy there! Prepare to embark on a fascinating journey as we set sail through the historic voyage of the Titanic. Welcome to Titanic Explorer, your one-stop app for diving deep into the legendary Titanic dataset and uncovering intriguing insights.
    """)
    st.subheader("🗄 Dataset")
    st.write("""
             Delve into the heart of the Titanic dataset, where you can explore the vast collection of passenger information. Learn about their age, class, gender, and more, and understand how these factors influenced their fate during that fateful journey.
             """)
    st.subheader("🔍 EDA (Exploratory Data Analysis)")
    st.write("""
             Uncover the hidden stories within the dataset through our comprehensive analysis. We've created interactive visualizations that provide a clear picture of trends, patterns, and correlations that existed on that ill-fated voyage.
             """)
    st.subheader("🔮 App for Prediction")
    st.write("""
             Ever wondered if you were aboard the Titanic, what would be your chances of survival? Our prediction section will predict your survival probability based on specific inputs. Explore various scenarios and see if you would've made it to safety.
             """)
    st.subheader("📏 Evaluation Metric")
    st.write("""
             Curious about how we measure the accuracy of our predictions? Our evaluation metric section will enlighten you on the methodologies and criteria we employ to ensure the precision and reliability of our predictions.
             """)
    st.subheader("📊 Graphs")
    st.write("""
             Visuals speak louder than words! In this segment, feast your eyes on a plethora of engaging charts and graphs that present a dynamic visualization of the Titanic dataset, making complex data more accessible and enjoyable.
             """)
    st.subheader("👤 About Us")
    st.write("""
             Get to know the passionate team behind Titanic Explorer. We are a dedicated group of data enthusiasts who are eager to share our knowledge and expertise with you.
             """)
    st.subheader("📞 Contact")
    st.write("""
             We value your feedback and inquiries. This section enables you to get in touch with us easily. Let us know what you think, and we'll be thrilled to assist you with any queries you may have.
             """)
    st.markdown("""<br></br>""", unsafe_allow_html= True)
    st.divider()
    st.write("""
             At Titanic Explorer, we believe that learning can be both informative and enjoyable. Our user-friendly yet professional approach aims to make your experience seamless and engaging, regardless of your expertise level. So, whether you're an aspiring data scientist, history enthusiast, or simply curious about the Titanic, our app is tailor-made for you!
             
             ⚓ Get ready to unravel the captivating stories hidden within the Titanic dataset. Come aboard and explore the past like never before with Titanic Explorer. Let's navigate history together! ⚓
             """)


elif tabs == "Dataset":
    st.title("Titanic Dataset")
    st.divider()
    st.dataframe(dataset, use_container_width = True, height = 725)  

elif tabs == "EDA":
    st.title("Exploratory Data Analysis")
    st.divider()
    
    head = dataset.head()
    tail = dataset.tail()
    correlation_matrix = dataset.corr()    
    check_null = dataset.isnull().sum()
    check_null = pd.Series(check_null, name = "Null_Value_Count")
    total_null = dataset.isnull().sum().sum()
    distinct_count = dataset.nunique()
    distinct_count = pd.Series(distinct_count, name = "Unique_Value_Count")
    descriptive_stats = dataset.describe()
    
    if st.session_state.page == 0:
        st.subheader("Data Head")
        st.write("jhuwhjhejhrejhrejhrehjrejrh")
        st.dataframe(head, use_container_width = True, height = 225)
        
        st.divider()
        
        # Creating columns for navigating to next page and previous page
        col3, col4 = st.columns([1, 6])
        with col3:
            st.button("Data Tail ➡", on_click = next_page)

        with col4:
            pass
        
    elif st.session_state.page == 1:
        st.subheader("Data Tail")
        st.write("Finding the tail of our dataset means we look at the bottom 5 values of our dataset. This is used in exploratory data analysis as a way to share insight on large datasets, what is happening at the extreme end of our data. In pandas, this is accomplished using the code below. ")
        # Using Column Design
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    tail = dataset.tail()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to finf the tail of the dataset.")
        st.dataframe(tail, use_container_width = True, height = 225)
        
        st.divider()
        
        # Creating columns for navigating to next page and previous page
        col3, col4 = st.columns([1, 7])
        with col3:
            st.button("⬅ Data Head", on_click = previous_page)

        with col4:
            st.button("Data Correlation Matrix ➡", on_click = next_page)
        
    elif st.session_state.page == 2:
        st.subheader("Data Correlation Matrix")
        st.write("jhuwhjhejhrejhrejhrehjrejrh")
        st.dataframe(correlation_matrix, use_container_width = True, height = 250)
        
        st.divider()
        
        # Creating columns for navigating to next page and previous page
        col3, col4 = st.columns([1, 7])
        with col3:
            st.button("⬅ Data Tail", on_click = previous_page)

        with col4:
            st.button("Null Value Count (Columns) ➡", on_click = next_page)
        
    elif st.session_state.page == 3:
        st.subheader("Null Value Count (Columns)")
        st.write("jhuwhjhejhrejhrejhrehjrejrh")
        st.dataframe(check_null, width = 250, height = 250)
        
        st.divider()
        
        # Creating columns for navigating to next page and previous page
        col3, col4 = st.columns([1.5, 6])
        with col3:
            st.button("⬅ Data Correlation Matrix", on_click = previous_page)

        with col4:
            st.button("Unique Values (Columns) ➡", on_click = next_page)
    
    elif st.session_state.page == 4:
        st.subheader("Unique Values (Columns)")
        st.write("jhuwhjhejhrejhrejhrehjrejrh")
        st.dataframe(distinct_count, width = 250, height = 250)
        
        st.divider()
        
        # Creating columns for navigating to next page and previous page
        col3, col4 = st.columns([1.5, 7])
        with col3:
            st.button("⬅ Null Value Count", on_click = previous_page)

        with col4:
            st.button("Data Descriptive Statistics ➡", on_click = next_page)
        
    elif st.session_state.page == 5:
        st.subheader("Data Descriptive Statistics")
        st.write("jhuwhjhejhrejhrejhrehjrejrh")
        st.dataframe(descriptive_stats, height = 325, use_container_width = True)
        
        st.divider()
        
        # Creating columns for navigating to next page and previous page
        col3, col4 = st.columns([1.5, 6])
        with col3:
            st.button("⬅ Unique Values (Columns)", on_click = previous_page)

        with col4:
            pass


elif tabs == "App":
    st.session_state['page'] = 3


elif tabs == "Evaluation":
    st.session_state['page'] = 4


elif tabs == "Graphs":
    st.session_state['page'] = 5


elif tabs == "About Us":
    st.session_state['page'] = 6


elif tabs == "Contact":
    st.session_state['page'] = 7

    
    
    



