import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# 📄 Load dataset (USE RAW STRING FOR WINDOWS PATH)
df = pd.read_json(r"C:\Users\SHYAM PRASATH\OneDrive\Desktop\Nigga\Prajan Project\News_Category_Dataset_v3.json", lines=True)
df = df[['headline', 'category']]

# 🔁 Simplify categories
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
    'WORLD NEWS': 'World',
    'COLLEGE': 'Education',
    'ARTS & CULTURE': 'Entertainment',
    'MEDIA': 'Media',
    'CRIME': 'Crime'
}

df['category'] = df['category'].map(category_map)
df = df.dropna()

# 🧮 Show category distribution
print("📊 Category distribution:\n")
print(df['category'].value_counts())

# 📂 Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['category'], test_size=0.2, random_state=42)

# 🔧 Model pipeline with Logistic Regression
model = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# 📈 Evaluate model
y_pred = model.predict(X_test)
print("\n📋 Model Performance (on test data):\n")
print(classification_report(y_test, y_pred))

# 🔮 User Input for Prediction
print("\n🔎 Automatic Topic Classifier")
while True:
    user_input = input("\nEnter a news headline (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    prediction = model.predict([user_input])[0]
    print(f"📌 Predicted Category: {prediction}")
