
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = """[PUT ANY TEXT FROM ANY WIKIPEDIA]"""

stopwords = list(STOP_WORDS)
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
tokens = [token.text for token in doc]
punctuation = punctuation + '\n'

word_frequencies = {}
for word in doc:
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
        	# If a word reached this condition, it means it's neither a stopword nor a punctuation
        	# then we check if it's already added to the `word_frequencies` dictionary
        	if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
			else:
            	word_frequencies[word.text] += 1


max_frequency = max(word_frequencies.values())


for word in word_frequencies.keys():
	word_frequencies[word] = word_frequencies[word]/max_frequency

sentence_tokens = [sentence for sentence in doc.sents]

sentence_scores = {}
# For every sentence in the paragraph, we sum up the score for all the words in it.
for sent in sentence_tokens:
	for word in sent:
    	if word.text.lower() in word_frequencies.keys():
        	if sent not in sentence_scores.keys():
            	sentence_scores[sent] = word_frequencies[word.text.lower()]
        	else:
            	sentence_scores[sent] += word_frequencies[word.text.lower()]



select_length = int(len(sentence_tokens)*0.3)

summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)

final_summary = [word.text for word in summary]

summary = ' '.join(final_summary)

print(summary)
