import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Title
st.title("📰 News Headline Topic Classifier")
st.markdown("Classifies news headlines into topics like Politics, Sports, Entertainment, etc.")

# Load and prepare data
@st.cache_data
def load_and_train_model():
    df = pd.read_json(r"C:\Users\SHYAM PRASATH\OneDrive\Desktop\Nigga\Prajan Project\News_Category_Dataset_v3.json", lines=True)
    df = df[['headline', 'category']]

    category_map = {
        'POLITICS': 'Politics',
        'WELLNESS': 'Health',
        'ENTERTAINMENT': 'Entertainment',
        'TRAVEL': 'Travel',
        'STYLE & BEAUTY': 'Lifestyle',
        'PARENTING': 'Lifestyle',
        'HEALTHY LIVING': 'Health',
        'QUEER VOICES': 'Social',
        'FOOD & DRINK': 'Food',
        'BUSINESS': 'Business',
        'SPORTS': 'Sports',
        'TECH': 'Technology',
        'SCIENCE': 'Science',
        'WORLD NEWS': 'World news',
        'COLLEGE': 'Education',
        'ARTS & CULTURE': 'Entertainment',
        'MEDIA': 'Media',
        'CRIME': 'Crime'
    }

    df['category'] = df['category'].map(category_map)
    df = df.dropna()

    X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['category'], test_size=0.2, random_state=42)

    model = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    return model

model = load_and_train_model()

# Input section
st.subheader("Enter a news headline below:")

user_input = st.text_input("Headline", "")

if user_input:
    prediction = model.predict([user_input])[0]
    st.success(f"📌 **Predicted Category:** {prediction}")
