import os
import pickle
import pyLDAvis.gensim
import gensim
import nltk
from nltk.corpus import wordnet as wn
from spacy.lang.en import English
from pathlib import Path
import logging
from gensim import corpora
import numpy as np


class topic_modelling:
    '''
    This class is used for identifying the topic of online media. Primarily designed for news articles, however, is also a genera purpose.
    '''

    en_stop = set(nltk.corpus.stopwords.words('english'))
    parser = English()

    script_dir = Path(__file__).parent
    models_dir = os.path.join(script_dir, "models")
    data_dir = os.path.join(script_dir, "data")

    def __init__(self):
        '''
        The constructor
        '''

        if not os.path.isdir(self.models_dir):
            os.mkdir(self.models_dir)

        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

    def _tokenize(self, text):
        '''
        Tokenise a given piece of text
        :param text:
        :return: list of lda tokens
        '''

        lda_tokens = []
        tokens = self.parser(text)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lda_tokens.append(token.lower_)
        return lda_tokens

    def _get_lemma(self, word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    def _prepare_text_for_lda(self, text):
        tokens = self._tokenize(text)
        tokens = [token for token in tokens if len(token) > 4]  # Removes words in list less than 5 characters
        tokens = [token for token in tokens if
                  token not in self.en_stop]  # Removes words in the list that are stop words
        tokens = [self._get_lemma(token) for token in
                  tokens]  # Performs stemming and Stemming and lemmatization on all words in list
        return tokens

    def re_train(self, number_of_topics=4, number_of_passes=15, dataset=os.path.join(data_dir, 'dataset.txt'),
                 dataset_encoding="utf-8"):
        '''
        Depending on the state of this library it may not come pre-trained.
        :param number_of_topics:
        :param number_of_passes:
        :param dataset:
        :param dataset_encoding:
        :return:
        '''

        text_data = []
        with open(dataset, encoding=dataset_encoding) as file:
            for line in file:
                tokens = self._prepare_text_for_lda(line)
                text_data.append(tokens)

        dictionary = corpora.Dictionary(
            text_data)  # Dictionary encapsulates the mapping between normalized words and their integer ids
        corpus = [dictionary.doc2bow(text) for text in
                  text_data]  # Converts each list of words (document) into a bag of words format

        pickle.dump(corpus, open(os.path.join(self.script_dir, os.path.join(self.models_dir, 'corpus.pkl')), 'wb'))
        dictionary.save(os.path.join(self.models_dir,
                                     'dictionary.gensim'))  # Saves to storage so large objects don't have to be stored in memory.

        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=number_of_topics, id2word=dictionary,
                                                   passes=number_of_passes)
        ldamodel.save(os.path.join(self.models_dir,
                                   'model.gensim'))  # Saves to storage so large objects don't have to be stored in memory.
        number_of_topics = ldamodel.print_topics(num_words=4)
        for topic in number_of_topics:
            logging.info(topic)

    def identify_topic(self, text):
        new_doc = self._prepare_text_for_lda(text)

        dictionary = gensim.corpora.Dictionary.load(os.path.join(self.models_dir, 'dictionary.gensim'))
        new_doc_bow = dictionary.doc2bow(new_doc)

        for item in new_doc_bow:

            # Check if files exist for training.
            if not os.path.isfile(os.path.join(self.models_dir, 'model.gensim')):
                raise Exception("Identifier not trained. Please train before using.")

            word = dictionary[item[0]]
            quantity = item[1]

            logging.info("The stem '{}' is used {} times".format(word, quantity))

        ldamodel = gensim.models.ldamodel.LdaModel.load(os.path.join(self.models_dir, 'model.gensim'))
        topics = ldamodel.get_document_topics(
            new_doc_bow)  # Here we use the LDA object we've trained and provide the new document to get it's topics - These probabilities add up to 1

        filtered_topics = []
        for topic_set in topics:
            topic = topic_set[0]
            topic_score = float(round(float(topic_set[1]), 6))
            filtered_topics.append((topic,topic_score))

        return filtered_topics

    def get_topics(self, number_of_words=4):

        # Check if files exist for training.
        if not os.path.isfile(os.path.join(self.models_dir, 'model.gensim')):
            raise Exception("Identifier not trained. Please train before using.")

        lda = gensim.models.ldamodel.LdaModel.load(os.path.join(self.models_dir, 'model.gensim'))

        return lda.print_topics(num_words=number_of_words)

    def lda_display(self):
        '''
        Displays the lda
        :return:
        '''

        # Check if files exist for training.
        if not os.path.isfile(os.path.join(self.models_dir, 'dictionary.gensim')):
            raise Exception("Identifier not trained. Please train before using.")

        dictionary = gensim.corpora.Dictionary.load(os.path.join(self.models_dir, 'dictionary.gensim'))
        corpus = pickle.load(open(os.path.join(self.models_dir, 'corpus.pkl'),
                                  'rb'))  # The corpus is the list of lists of each document's bows
        lda = gensim.models.ldamodel.LdaModel.load(os.path.join(self.models_dir, 'model.gensim'))

        lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=True)
        pyLDAvis.display(lda_display)

    def get_bag_of_words(self, text):
        '''
        Performs bag of words on a given piece of text.
        :param text:
        '''

        # Check if files exist for training.
        if not os.path.isfile(os.path.join(self.models_dir, 'dictionary.gensim')):
            raise Exception("Identifier not trained. Please train before using.")

        dictionary = gensim.corpora.Dictionary.load(os.path.join(self.models_dir, 'dictionary.gensim'))

        tokens = self._prepare_text_for_lda(text)
        return dictionary.doc2bow(tokens)

    def get_dictionary(self):
        '''
        Returns the dictionary used in modelling
        '''

        # Check if files exist for training.
        if not os.path.isfile(os.path.join(self.models_dir, 'dictionary.gensim')):
            raise Exception("Identifier not trained. Please train before using.")

        return gensim.corpora.Dictionary.load(os.path.join(self.models_dir, 'dictionary.gensim'))

    def get_corpus(self):
        '''
        Returns the corpus
        '''

        # Check if files exist for training.
        if not os.path.isfile(os.path.join(self.models_dir, 'corpus.pkl')):
            raise Exception("Identifier not trained. Please train before using.")

        return pickle.load(open(os.path.join(self.models_dir, 'corpus.pkl'), 'rb'))

    def get_lda_model(self):
        '''
        Returns the lda model
        '''

        # Check if files exist for training.
        if not os.path.isfile(os.path.join(self.models_dir, 'model.gensim')):
            raise Exception("Identifier not trained. Please train before using.")

        return gensim.models.ldamodel.LdaModel.load(os.path.join(self.models_dir, 'model.gensim'))
