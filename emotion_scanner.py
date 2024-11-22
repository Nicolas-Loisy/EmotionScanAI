import pandas as pd
from transformers import pipeline

def load_models():
    """
    Charge les modèles de classification pour l'analyse d'émotions et de sentiments.
    
    Retourne :
        tuple: Deux pipelines, l'un pour les émotions et l'autre pour les sentiments.
    """
    emotion_model = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
    sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    emotion_pipe = pipeline("text-classification", model=emotion_model, return_all_scores=True)
    sentiment_pipe = pipeline("text-classification", model=sentiment_model, return_all_scores=True)
    
    return emotion_pipe, sentiment_pipe

def analyze_tweet(tweet, emotion_pipe, sentiment_pipe):
    """
    Analyse un tweet pour prédire les émotions et les sentiments.
    
    Args:
        tweet (str): Le texte du tweet.
        emotion_pipe (Pipeline): Pipeline pour l'analyse des émotions.
        sentiment_pipe (Pipeline): Pipeline pour l'analyse des sentiments.
    
    Retourne :
        tuple: Les émotions et le sentiment principal sous forme de chaînes.
    """
    # Analyse des émotions
    emotion_result = emotion_pipe(tweet)
    primary_emotion = max(emotion_result[0], key=lambda x: x['score'])['label']
    
    # Analyse des sentiments
    sentiment_result = sentiment_pipe(tweet)
    primary_sentiment = max(sentiment_result[0], key=lambda x: x['score'])['label']
    
    return primary_emotion, primary_sentiment

def process_csv(input_file, output_file, emotion_pipe, sentiment_pipe):
    """
    Traite un fichier CSV contenant des tweets pour y ajouter les colonnes 'emotion' et 'sentiment'.
    
    Args:
        input_file (str): Chemin du fichier CSV en entrée.
        output_file (str): Chemin du fichier CSV en sortie.
        emotion_pipe (Pipeline): Pipeline pour l'analyse des émotions.
        sentiment_pipe (Pipeline): Pipeline pour l'analyse des sentiments.
    """
    # Chargement des données
    df = pd.read_csv(input_file)
    
    if 'tweet' not in df.columns:
        raise ValueError("Le fichier CSV doit contenir une colonne 'tweet'.")
    
    # Initialisation des colonnes pour les résultats
    df['emotion'] = None
    df['sentiment'] = None
    
    # Analyse de chaque tweet
    for index, row in df.iterrows():
        tweet = row['tweet']
        emotion, sentiment = analyze_tweet(tweet, emotion_pipe, sentiment_pipe)
        df.at[index, 'emotion'] = emotion
        df.at[index, 'sentiment'] = sentiment
    
    # Sauvegarde du fichier modifié
    df.to_csv(output_file, index=False)
    print(f"Fichier sauvegardé avec succès : {output_file}")

def main(input_file, output_file):
    """
    Fonction principale pour analyser les tweets d'un fichier CSV et sauvegarder les résultats.
    
    Args:
        input_file (str): Chemin du fichier CSV en entrée.
        output_file (str): Chemin du fichier CSV en sortie.
    """
    print("Chargement des modèles...")
    emotion_pipe, sentiment_pipe = load_models()
    
    print("Traitement du fichier CSV...")
    process_csv(input_file, output_file, emotion_pipe, sentiment_pipe)
    print("Analyse terminée.")

if __name__ == "__main__":
    import argparse
    
    # Configuration des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Analyse des émotions et sentiments des tweets dans un fichier CSV.")
    parser.add_argument("input_file", type=str, help="Chemin du fichier CSV en entrée.")
    parser.add_argument("output_file", type=str, help="Chemin du fichier CSV en sortie.")
    
    args = parser.parse_args()
    
    # Exécution du script
    main(args.input_file, args.output_file)
