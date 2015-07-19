from nltk import *


my_text = "The coolest job in the next 10 years will be statisticians. People think I'm joking, but who would've " \
          "guessed that computer engineers would've been the coolest job of the 1990s?"
nltk_tokens = nltk.word_tokenize(my_text)
print nltk_tokens
print
stemmer = LancasterStemmer()
print [stemmer.stem(word) for word in nltk_tokens]
print
print nltk.pos_tag(nltk_tokens)
print
# Named Entity Recognition
text = "Elvis Aaron Presley was an American singer and actor. Born in Tupelo, Mississippi, when Presley was 13 years " \
       "old he and his family relocated to Memphis, Tennessee."
chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
print chunks