import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_data(file_path):
    df = pd.read_csv(file_path, encoding="utf8", sep=";")
    return df

def train_model(df, text_column, label_column):
    x = df[text_column].astype(str).fillna("")
    y = df[label_column].astype(str).fillna("")
    
    vectorizer = TfidfVectorizer(max_features=1000)
    x_train_tfidf = vectorizer.fit_transform(x)
    
    model = LogisticRegression(max_iter=200)
    model.fit(x_train_tfidf, y)
    
    return model, vectorizer

def classify_tweets(tweets, model, vectorizer):
    tweets_tfidf = vectorizer.transform(tweets)
    predictions = model.predict(tweets_tfidf)
    return predictions

def main(training_file, input_file, output_file):
    """
    Fonction principale pour analyser les tweets d'un fichier CSV et sauvegarder les résultats.
    
    Args:
        training_file (str): Chemin du fichier CSV pour l'entraînement.
        input_file (str): Chemin du fichier CSV en entrée.
        output_file (str): Chemin du fichier CSV en sortie.
    """
    df_train = load_data(training_file)
    emotion_model, emotion_vectorizer = train_model(df_train, 'Text', 'emotion')
    sentiment_model, sentiment_vectorizer = train_model(df_train, 'Text', 'sentiment')
    
    df_input = load_data(input_file)
    tweets = df_input['tweet'].tolist()
    predicted_emotions = classify_tweets(tweets, emotion_model, emotion_vectorizer)
    predicted_sentiments = classify_tweets(tweets, sentiment_model, sentiment_vectorizer)
    
    df_input['emotion'] = predicted_emotions
    df_input['sentiment'] = predicted_sentiments
    
    print("Création du fichier CSV mis à jour...")
    df_input.to_csv(output_file, index=False)
    print("Analyse terminée.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse des émotions et des sentiments des tweets dans un fichier CSV.")
    parser.add_argument("training_file", type=str, help="Chemin du fichier CSV pour l'entraînement.")
    parser.add_argument("input_file", type=str, help="Chemin du fichier CSV en entrée.")
    parser.add_argument("output_file", type=str, help="Chemin du fichier CSV en sortie.")
    
    args = parser.parse_args()
    
    main(args.training_file, args.input_file, args.output_file)