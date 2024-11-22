# EmotionScanAI

Ce projet analyse les émotions et les sentiments des tweets en utilisant des modèles de classification basés sur transformers.

## Prérequis

- Python 3.12.2 ou supérieur
- `pip` pour installer les dépendances

## Installation

1. Clonez le dépôt :

   ```sh
   git clone <URL_DU_DEPOT>
   cd EmotionScanAI
   ```

2. Installez les dépendances :

   ```sh
   pip install -r requirements.txt
   ```

## Utilisation

Pour analyser les émotions et les sentiments des tweets dans un fichier CSV, exécutez la commande suivante :

```sh
python emotion_scanner.py data/Tweets.csv data/output_tweets.csv
```

## Structure du projet

- emotion_scanner.py : Script principal pour analyser les tweets.
- data : Contient les fichiers CSV d'entrée et de sortie.
- Notebook : Contient les notebooks Jupyter pour l'expérimentation et le développement.

## Exemple de fichier CSV d'entrée (`data/Tweets.csv`)

```csv
tweet
I bet everything will work out in the end :)
I'm feeling very disappointed with the situation.
The world is a beautiful place!
I am so angry about what happened.
```

## Exemple de fichier CSV de sortie (`data/output_tweets.csv`)

```csv
tweet,emotion,sentiment
I bet everything will work out in the end :),optimism,positive
I'm feeling very disappointed with the situation.,sadness,negative
The world is a beautiful place!,joy,positive
I am so angry about what happened.,anger,negative
```

## Auteurs

- Nicolas Loisy
