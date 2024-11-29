import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_data(file_path):
    df = pd.read_csv(file_path, encoding="utf8", sep=";")
    return df

def train_emotion_model(training_file):
    df = load_data(training_file)
    x = df['Text'].astype(str).fillna("")
    y = df['emotion'].astype(str).fillna("")
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=1000)
    x_train_tfidf = vectorizer.fit_transform(x)
    
    model = LogisticRegression(max_iter=200)
    model.fit(x_train_tfidf, y)
    
    return model, vectorizer

def train_sentiment_model(training_file):
    df = load_data(training_file)
    x = df['Text'].astype(str).fillna("")
    y = df['sentiment'].astype(str).fillna("")
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=1000)
    x_train_tfidf = vectorizer.fit_transform(x)
    
    model = LogisticRegression(max_iter=200)
    model.fit(x_train_tfidf, y)
    
    return model, vectorizer

def classify_emotions(tweets, model, vectorizer):
    tweets_tfidf = vectorizer.transform(tweets)
    predicted_emotions = model.predict(tweets_tfidf)
    return predicted_emotions

def classify_sentiments(tweets, model, vectorizer):
    tweets_tfidf = vectorizer.transform(tweets)
    predicted_sentiments = model.predict(tweets_tfidf)
    return predicted_sentiments

def main(training_file, input_file, output_file):
    """
    Fonction principale pour analyser les tweets d'un fichier CSV et sauvegarder les résultats.
    
    Args:
        training_file (str): Chemin du fichier CSV pour l'entraînement.
        input_file (str): Chemin du fichier CSV en entrée.
        output_file (str): Chemin du fichier CSV en sortie.
    """
    emotion_model, emotion_vectorizer = train_emotion_model(training_file)
    sentiment_model, sentiment_vectorizer = train_sentiment_model(training_file)
    
    df = load_data(input_file)
    tweets = df['tweet'].tolist()
    predicted_emotions = classify_emotions(tweets, emotion_model, emotion_vectorizer)
    predicted_sentiments = classify_sentiments(tweets, sentiment_model, sentiment_vectorizer)
    
    df['emotion'] = predicted_emotions
    df['sentiment'] = predicted_sentiments
    
    print("Création du fichier CSV mis à jour...")
    df.to_csv(output_file, index=False)
    print("Analyse terminée.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse des émotions et des sentiments des tweets dans un fichier CSV.")
    parser.add_argument("training_file", type=str, help="Chemin du fichier CSV pour l'entraînement.")
    parser.add_argument("input_file", type=str, help="Chemin du fichier CSV en entrée.")
    parser.add_argument("output_file", type=str, help="Chemin du fichier CSV en sortie.")
    
    args = parser.parse_args()
    
    main(args.training_file, args.input_file, args.output_file)