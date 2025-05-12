# 📰 News Headline Topic Classifier

The **News Headline Topic Classifier** is an interactive web application built using **Streamlit** that classifies news headlines into various predefined categories such as **Politics**, **Sports**, **Entertainment**, **Health**, **Technology**, and more. The project leverages machine learning techniques to process text data and predict the category of a news headline.

## 🚀 Features

- **Real-time Classification**: Classifies news headlines into categories like Politics, Sports, Technology, etc.
- **Interactive Web Application**: Built with **Streamlit**, making it user-friendly and interactive.
- **Text Processing**: Utilizes **TF-IDF Vectorizer** for transforming text data and **Logistic Regression** for classification.
- **Predefined Categories**: Supports categories such as **Politics**, **Health**, **Entertainment**, **Sports**, **Business**, **Technology**, and many more.

## 🔨 Technologies Used

- **Streamlit**: A Python library for creating interactive web applications.
- **Pandas**: Data manipulation and analysis library.
- **Scikit-learn**: Machine learning library for building and evaluating models.
  - **Logistic Regression**: Used for text classification.
  - **TfidfVectorizer**: Converts the text data into numerical features.
- **Python**: Programming language used to build the application.

## 🧑‍💻 Setup Instructions

1. **Clone the Repository**:
   Clone this repository to your local machine.
   
   ```bash
   git clone https://github.com/Shyam-Prasath/news-headline-classifier.git
   ```

2. **Install Dependencies**:
   Install the necessary libraries.

   ```bash
   pip install pandas scikit-learn streamlit
   ```

3. **Prepare the Dataset**:
   Ensure you have the dataset `News_Category_Dataset_v3.json` (available from various sources or can be downloaded from Kaggle).
   Place the dataset in the appropriate directory or change the path in the code to where the dataset is located.

4. **Run the Application**:
   Start the Streamlit app by running the following command:

   ```bash
   streamlit run app.py
   ```

   This will launch a local web server where you can interact with the application.

## 🧠 How It Works

- **Data Preprocessing**: The dataset is loaded and preprocessed by mapping category names to more readable labels. Missing values are dropped, and data is split into training and testing sets.
- **Model Training**: A machine learning pipeline is built using TF-IDF Vectorizer for transforming text data and Logistic Regression for classification. The model is trained on the news headlines and categories.
- **Prediction**: Once the model is trained, the user can input a news headline into the app, and the model will predict its category.

## 🔍 Example

**Input**:
```
"Apple unveils new iPhone with groundbreaking features"
```

**Output**:
```
📌 Predicted Category: Technology
```
