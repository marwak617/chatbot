import nltk
# Uncomment the following line if you haven't downloaded the necessary resources
# nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
import numpy as np

# Initialize the French stemmer
stemmer = SnowballStemmer("french")

# Tokenize a sentence into words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stem a word to its root form
def stem(word):
    return stemmer.stem(word.lower())  # Convert to lowercase before stemming

# Create a bag of words representation
def bag_of_words(tokenized_sentence, all_words):
    # Stem each word in the tokenized sentence
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    # Initialize a zero array for the bag of words
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    # Mark the index as 1.0 if the word exists in the tokenized_sentence
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

# Example usage
sentence = "L'ENSA ne dispose pas d'une résidence universitaire."
words = ["ensa", "dispose", "resid", "universitair", "hello", "bye", "merci"]
tokenized = tokenize(sentence)
bag = bag_of_words(tokenized, words)

print(f"Tokenized sentence: {tokenized}")
print(f"Bag of words: {bag}")
