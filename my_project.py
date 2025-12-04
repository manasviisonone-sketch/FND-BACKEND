import pandas as pd
import os
import joblib
import re
import nltk
from nltk.corpus import stopwords
import traceback  # Import traceback at the top
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report # For the real score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

# -----------------------------------------------------------------
# STEP 1: PREPARE TEXT CLEANING FUNCTION
# -----------------------------------------------------------------
print("Setting up text cleaning function...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords'...")
    nltk.download('stopwords')
    
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    clean_words = [word for word in words if word not in stop_words]
    text = ' '.join(clean_words)
    return text

print("--- Starting Model Training Pipeline ---")

# This try block covers the entire script
try:
    # -----------------------------------------------------------------
    # STEP 2: LOAD AND COMBINE THE DATASETS
    # -----------------------------------------------------------------

    # --- 1. Load your OLD 56,000-article dataset ---
    print("Loading final_dataset.csv (Old Political Data)...")
    df_old = pd.read_csv("final_dataset.csv")
    df_old = df_old.dropna(subset=['text', 'label']) # Drop empty rows
    df_old['label'] = df_old['label'].map({'real': 0, 'fake': 1})
    df_old = df_old[['text', 'label']]
    print(f"Loaded {len(df_old)} old articles.")

    # --- 2. Load your NEW satire dataset ---
    print("Loading Onion.csv (New Satire Data)...")
    try:
        # --- THIS IS THE BRUTE FORCE FIX ---
        # Using the full absolute path to bypass the OneDrive bug
        file_path = r"C:\Users\sonone\OneDrive\Desktop\fakenewsdetector backend final\Onion.csv"
        df_new = pd.read_csv(file_path) 
        
    except FileNotFoundError:
        print("\n--- FATAL ERROR ---")
        print(f"Could not find file at path: {file_path}")
        print("Make sure this path is 100% correct in your File Explorer.")
        exit()
        
    df_new = df_new.dropna(subset=['text', 'label']) # Drop empty rows
    df_new = df_new[['text', 'label']]
    print(f"Loaded {len(df_new)} new satire articles.")

    # --- 3. Combine them into one giant dataframe ---
    print("Combining datasets...")
    df = pd.concat([df_old, df_new], ignore_index=True)

    # --- 4. Shuffle the combined dataset (VERY IMPORTANT!) ---
    df = df.sample(frac=1).reset_index(drop=True)

    print(f"--- Total combined dataset size: {len(df)} articles ---")
    print("Class distribution (before SMOTE):")
    print(df['label'].value_counts())

    print("Cleaning all combined text data... (this may take a minute)")
    df['text'] = df['text'].apply(clean_text)
    print("Text cleaning complete.")

    # Define our X (features) and y (target)
    X = df['text']  # X is now the cleaned text
    y = df['label'].astype(int) # Ensure labels are integers


    # -----------------------------------------------------------------
    # STEP 3: TF-IDF VECTORIZATION (for classic ML models)
    # -----------------------------------------------------------------
    print("\nCreating TF-IDF features from clean text...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("TF-IDF Vectorizer saved to 'tfidf_vectorizer.pkl'")


    # -----------------------------------------------------------------
    # STEP 4: SPLIT DATA AND APPLY SMOTE (for classic ML models)
    # -----------------------------------------------------------------
    print("Splitting data for classic ML models...")
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Original training shape: {X_train_tfidf.shape}")
    print("Applying SMOTE to training data...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_tfidf, y_train)
    
    print(f"New balanced training shape: {X_train_res.shape}")


    # -----------------------------------------------------------------
    # STEP 5: DATA PREPARATION (for LSTM model)
    # -----------------------------------------------------------------
    print("\nSplitting and preparing data for LSTM...")
    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    max_words = 10000
    max_len = 150 

    print("Tokenizing clean text...")
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train_text)

    joblib.dump(tokenizer, 'lstm_tokenizer.pkl')
    print("LSTM Tokenizer saved to 'lstm_tokenizer.pkl'")

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    y_train_cat = to_categorical(y_train_text, 2)
    y_test_cat = to_categorical(y_test_text, 2)


    # -----------------------------------------------------------------
    # MODEL 1: LSTM
    # -----------------------------------------------------------------
    print("\n--- Training Model 1: LSTM ---")
    model = Sequential()
    model.add(Embedding(max_words, 64, input_length=max_len))
    model.add(SpatialDropout1D(0.2)) 
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train_pad, y_train_cat, 
              batch_size=32, 
              epochs=3, 
              validation_split=0.1,
              verbose=1) 
              
    loss, accuracy = model.evaluate(X_test_pad, y_test_cat)
    print("-----------------------------------------")
    print(f"SUCCESS! The LSTM model is {accuracy * 100:.2f}% accurate!")
    print("-----------------------------------------")
    model.save('lstm_model.keras')
    print("LSTM Model Saved!")


    # -----------------------------------------------------------------
    # MODEL 2: NAIVE BAYES
    # -----------------------------------------------------------------
    print("\n--- Training Model 2: Naive Bayes ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_res, y_train_res)
    
    nb_predictions = nb_model.predict(X_test_tfidf)
    nb_score = metrics.accuracy_score(y_test, nb_predictions)
    print("-----------------------------------------")
    print(f"SUCCESS! The Naive Bayes model is {nb_score * 100:.2f}% accurate!")
    print(classification_report(y_test, nb_predictions, target_names=['REAL (0)', 'FAKE (1)']))
    print("-----------------------------------------")
    joblib.dump(nb_model, 'nb_model.pkl')


    # -----------------------------------------------------------------
    # MODEL 3: LOGISTIC REGRESSION
    # -----------------------------------------------------------------
    print("\n--- Training Model 3: Logistic Regression ---")
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_train_res, y_train_res)
    
    logreg_predictions = logreg_model.predict(X_test_tfidf)
    logreg_score = metrics.accuracy_score(y_test, logreg_predictions)
    print("-----------------------------------------")
    print(f"SUCCESS! The Logistic Regression model is {logreg_score * 100:.2f}% accurate!")
    print(classification_report(y_test, logreg_predictions, target_names=['REAL (0)', 'FAKE (1)']))
    print("-----------------------------------------")
    joblib.dump(logreg_model, 'logreg_model.pkl')


    # -----------------------------------------------------------------
    # MODEL 4: SUPPORT VECTOR MACHINE (SVM)
    # -----------------------------------------------------------------
    print("\n--- Training Model 4: SVM ---")
    # --- THIS IS THE FIXED TYPO ---
    svm_model = SVC(probability=True) 
    svm_model.fit(X_train_res, y_train_res)
    
    svm_predictions = svm_model.predict(X_test_tfidf)
    svm_score = metrics.accuracy_score(y_test, svm_predictions)
    print("-----------------------------------------")
    print(f"SUCCESS! The SVM model is {svm_score * 100:.2f}% accurate!")
    print(classification_report(y_test, svm_predictions, target_names=['REAL (0)', 'FAKE (1)']))
    print("-----------------------------------------")
    joblib.dump(svm_model, 'svm_model.pkl')


    # -----------------------------------------------------------------
    # MODEL 5: K-NEAREST NEIGHBORS (KNN)
    # -----------------------------------------------------------------
    print("\n--- Training Model 5: KNN ---")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_res, y_train_res)
    
    knn_predictions = knn_model.predict(X_test_tfidf)
    knn_score = metrics.accuracy_score(y_test, knn_predictions)
    print("-----------------------------------------")
    print(f"SUCCESS! The KNN model is {knn_score * 100:.2f}% accurate!")
    print(classification_report(y_test, knn_predictions, target_names=['REAL (0)', 'FAKE (1)']))
    print("-----------------------------------------")
    joblib.dump(knn_model, 'knn_model.pkl')


    # -----------------------------------------------------------------
    # MODEL 6: NEURAL NETWORK (MLP)
    # -----------------------------------------------------------------
    print("\n--- Training Model 6: Neural Network (MLP) ---")
    nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    nn_model.fit(X_train_res, y_train_res)
    
    nn_predictions = nn_model.predict(X_test_tfidf)
    nn_score = metrics.accuracy_score(y_test, nn_predictions)
    print("-----------------------------------------")
    print(f"SUCCESS! The Neural Network model is {nn_score * 100:.2f}% accurate!")
    print(classification_report(y_test, nn_predictions, target_names=['REAL (0)', 'FAKE (1)']))
    print("-----------------------------------------")
    joblib.dump(nn_model, 'nn_model.pkl')
    
    print("\n--- ALL MODELS TRAINED AND SAVED SUCCESSFULLY! ---")

# -----------------------------------------------------------------
# FINAL 'EXCEPT' BLOCK
# THIS MUST HAVE ZERO INDENTATION (BE ALL THE WAY TO THE LEFT)
# -----------------------------------------------------------------
except Exception as e:
    print(f"\n--- AN ERROR OCCURRED ---")
    print(f"Error: {e}")
    traceback.print_exc()