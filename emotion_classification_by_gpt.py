from pydantic import BaseModel
from enum import Enum
from openai import OpenAI
import pandas as pd

# Initialiser le client OpenAI
client = OpenAI(api_key="OPENAI_API_KEY")

# Définir des énumérations pour les valeurs possibles
class EmotionEnum(str, Enum):
    anger = "anger"
    anticipation = "anticipation"
    disgust = "disgust"
    fear = "fear"
    joy = "joy"
    nan = "nan"
    optimism = "optimism"
    sadness = "sadness"
    surprise = "surprise"

class SentimentEnum(str, Enum):
    neutral = "neutral"
    positive = "positive"
    negative = "negative"

# Définir la structure pour les résultats classifiés avec des énumérations
class TweetClassification(BaseModel):
    tweet: str
    emotion: EmotionEnum
    sentiment: SentimentEnum

# Fonction pour classer les tweets
def classify_tweets(file_path: str):
    # Charger les tweets depuis un fichier CSV
    df = pd.read_csv(file_path)

    # Préparer les résultats structurés
    results = []

    # Parcourir les tweets et classer chaque tweet
    for tweet in df['tweet']:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Classify the emotion and sentiment of the tweet. "
                    "Emotion possible values: anger, anticipation, disgust, fear, joy, nan, optimism, sadness, surprise. "
                    "Sentiment possible values: neutral, positive, negative."
                )},
                {"role": "user", "content": f"Tweet: {tweet}"}
            ],
            response_format=TweetClassification
        )

        # Extraire les informations classifiées
        classified = completion.choices[0].message.parsed
        results.append({
            "tweet": classified.tweet,
            "emotion": classified.emotion,
            "sentiment": classified.sentiment
        })

    # Convertir les résultats en un DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def main(input_file, output_file):
    """
    Fonction principale pour analyser les tweets d'un fichier CSV et sauvegarder les résultats.
    
    Args:
        input_file (str): Chemin du fichier CSV en entrée.
        output_file (str): Chemin du fichier CSV en sortie.
    """
    # Classifier les tweets
    classified_df = classify_tweets(input_file)
    
    print("Traitement du fichier CSV...")
    # Sauvegarder les résultats dans un nouveau fichier CSV
    classified_df.to_csv(output_file, index=False)
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
