
# coding: utf-8

# 
# # Machine Translation Project
# This is a project from the natural language processing nanodegree from udacity
# 
# (https://br.udacity.com/course/natural-language-processing-nanodegree--nd892)
# 
# ## Introduction
# In this notebook, we will build a deep neural network that functions as part of an end-to-end machine translation pipeline. Your completed pipeline will accept English text as input and return the French translation.
# 
# - **Preprocess** - It'll convert text to sequence of integers.
# - **Models** Creates models which accepts a sequence of integers as input and returns a probability distribution over possible translations. We will try some basic types of neural networks that are often used for machine translation, in the end will build based on the lessons learnes your own model!
# - **Prediction** In the end the model will run on some sample English text.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('aimport', 'helper, tests')
get_ipython().run_line_magic('autoreload', '1')


# In[2]:


import collections

import helper
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Add, Reshape#, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


# ### Verify access to the GPU

# In[3]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# ## Dataset
# We begin by investigating the dataset that will be used to train and evaluate your pipeline.  The most common datasets used for machine translation are from [WMT](http://www.statmt.org/).  However, that will take a long time to train a neural network on.  We'll be using a special dataset for this project that contains a small vocabulary.  We'll be able to train our model in a more reasonable time with this dataset.
# ### Load Data
# The data is located in `data/small_vocab_en` and `data/small_vocab_fr`. The `small_vocab_en` file contains English sentences with their French translations in the `small_vocab_fr` file. Load the English and French data from these files from running the cell below.

# In[4]:


# Load English data
english_sentences = helper.load_data('data/small_vocab_en')
# Load French data
french_sentences = helper.load_data('data/small_vocab_fr')

print('Dataset Loaded')


# ### Files
# Each line in `small_vocab_en` contains an English sentence with the respective translation in each line of `small_vocab_fr`.  View the first two lines from each file.

# In[5]:


for sample_i in range(2):
    print('small_vocab_en Line {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
    print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, french_sentences[sample_i]))


# From looking at the sentences, you can see they have been preprocessed already.  The puncuations have been delimited using spaces. All the text have been converted to lowercase.  This saves a little work right now, but the text requires more preprocessing.
# ### Vocabulary
# The complexity of the problem is determined by the complexity of the vocabulary.  A more complex vocabulary is a more complex problem.  Let's look at the complexity of the dataset we'll be working with.

# In[6]:


english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')


# For comparison, _Alice's Adventures in Wonderland_ contains 2,766 unique words of a total of 15,500 words.
# ## Preprocess
# For this project, we won't use text data as input to your model. Instead, we'll convert the text into sequences of integers using the following preprocess methods:
# 1. Tokenize the words into ids
# 2. Add padding to make all the sequences the same length.
# 
# ### Tokenize
# 
# Turning each sentence into a sequence of words ids using Keras's [`Tokenizer`](https://keras.io/preprocessing/text/#tokenizer) function. Use this function to tokenize `english_sentences` and `french_sentences` in the cell below.
# 
# Running the cell will run `tokenize` on sample data and show output for debugging.

# In[7]:


from keras.preprocessing.text import text_to_word_sequence

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    
    my_tokenizer = Tokenizer()
    my_tokenizer.fit_on_texts(x)
    sequences = my_tokenizer.texts_to_sequences(x)
    
    return sequences, my_tokenizer


# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))


# ### Padding
# When batching the sequence of word ids together, each sequence needs to be the same length.  Since sentences are dynamic in length, we can add padding to the end of the sequences to make them the same length.
# 
# Using Keras's [`pad_sequences`](https://keras.io/preprocessing/sequence/#pad_sequences) function.

# In[8]:


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    if length == None:
        maxlen = max([len(sentence) for sentence in x])
    else:
        maxlen = length
    
    padded = pad_sequences(x, maxlen=maxlen, padding='post', truncating='post')

    return np.asarray(padded)


# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))


# ### Preprocess Pipeline

# In[9]:


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =    preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)


# ## Models
# In this section, we will experiment with various neural network architectures.
# We will begin by training four relatively simple architectures.
# - Model 1 is a simple RNN
# - Model 2 is a RNN with Embedding
# - Model 3 is a Bidirectional RNN
# - Model 4 is an Encoder-Decoder RNN
# 
# After experimenting with the four simple architectures, we will construct a deeper architecture that should hopefully be based on all the lessons learned
# ### Ids Back to Text
# The neural network will be translating the input to words ids, which isn't the final form we want.  We want the French translation.  The function `logits_to_text` will bridge the gab between the logits from the neural network to the French translation. You'll be using this function to better understand the output of the neural network.

# In[10]:


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')


# ### Model 1: RNN
# ![RNN](images/rnn.png)
# A basic RNN model is a good baseline for sequence data.

# In[11]:


french_vocab_size = french_vocab_size + 1
english_vocab_size = english_vocab_size + 1


# In[12]:


def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    learning_rate = 0.001
    
    # Build the layers    
    i = Input(shape=input_shape[1:])
    g = GRU(64, return_sequences=True)(i)
    l = TimeDistributed(Dense((french_vocab_size)))(g)
    a = Activation('softmax')(l)
    model = Model(i, a)
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model

# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)
simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))


# ### Model 2: Embedding
# ![RNN](images/embedding.png)
# We've turned the words into ids, but there's a better representation of a word called word embeddings.  An embedding is a vector representation of the word that is close to similar words in n-dimensional space, where the n represents the size of the embedding vectors.
# 
# In this model, we'll create a RNN model using embedding.

# In[13]:


def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Implement
    learning_rate = 0.001

    i = Input(shape = input_shape[1:])
    e = Embedding(english_vocab_size, 32, input_length = output_sequence_length)(i)
    g = GRU(64, return_sequences=True)(e)
    t = TimeDistributed(Dense((french_vocab_size)))(g)
    a = Activation('softmax')(t)
    model = Model(i, a)
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model


# Reshape the input
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
#tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
embed_model = embed_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)

embed_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(embed_model.predict(tmp_x[:1])[0], french_tokenizer))


# ### Model 3: Bidirectional RNNs
# ![RNN](images/bidirectional.png)
# One restriction of a RNN is that it can't see the future input, only the past.  This is where bidirectional recurrent neural networks come in.  They are able to see the future data.

# In[14]:


def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Implement
    learning_rate = 0.001

    i = Input(shape = input_shape[1:])
    
    #We leave the embedding layer out on purpose this time to focus on the effects of the bidirectional model
    #e = Embedding(english_vocab_size, 32, input_length = output_sequence_length)(i)

    g = Bidirectional(GRU(64, return_sequences=True))(i)
    t = TimeDistributed(Dense((french_vocab_size)))(g)
    a = Activation('softmax')(t)
    model = Model(i, a)
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model


# Train and Print prediction(s)

# Reshape the input
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
bd_model = bd_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)

bd_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(bd_model.predict(tmp_x[:1])[0], french_tokenizer))


# ### Model 4: Encoder-Decoder 
# Time to look at encoder-decoder models.  This model is made up of an encoder and decoder. The encoder creates a matrix representation of the sentence.  The decoder takes this matrix as input and predicts the translation as output.

# In[15]:


def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Implement
    learning_rate = 0.002

    #ENCODER
    i_e = Input(shape = input_shape[1:])
    e_e = Embedding(english_vocab_size, 32, input_length = output_sequence_length)(i_e)
    g_e = GRU(64, return_sequences=True)(e_e)
    
    #DECODER
    #i_d = Input(shape = g_e.shape)(g_e)
    g_d = Bidirectional(GRU(64, return_sequences=True))(g_e)
    #g_d = Bidirectional((GRU(64, return_sequences=True)(i_d)), merge_mode='sum')
    
    t_d = TimeDistributed(Dense((french_vocab_size)))(g_d)
    a_d = Activation('softmax')(t_d)
    model = Model(i_e, a_d)
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model


# Reshape the input
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
#tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
encdec_model = encdec_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)

encdec_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(encdec_model.predict(tmp_x[:1])[0], french_tokenizer))


# ### Model 5: Custom 
# Now using everything we learned from the previous models do far to create a model that incorporates embedding and a bidirectional rnn into one model.

# In[23]:


def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    learning_rate = 0.005

    #ENCODER
    i_e = Input(shape = input_shape[1:])
    e_e = Embedding(english_vocab_size, 32, input_length = output_sequence_length)(i_e) #, input_length = output_sequence_length
    g_e = GRU(64, return_sequences=True)(e_e)
    #r_e = RepeatVector(output_sequence_length)(g_e)
    
    #DECODER
    g_d = Bidirectional(GRU(256, return_sequences=True))(g_e)  
    t_d = TimeDistributed(Dense((french_vocab_size)))(g_d)
    a_d = Activation('softmax')(t_d)
    model = Model(i_e, a_d)
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    '''
    #The Sequential writing 
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
    '''
    
    return model


print('Final Model Loaded')
# Train the final model

# Reshape the input
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)

# Train the neural network
m_f = model_final(tmp_x.shape,max_french_sequence_length,english_vocab_size,french_vocab_size)
m_f.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(m_f.predict(tmp_x[:1])[0], french_tokenizer))


# ## Prediction 

# In[24]:


def final_predictions(x, y, x_tk, y_tk):
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    """
    # Train neural network using model_final
    
    #Padding input
    max_french = max([len(i) for i in y])
    x = pad(x, max_french)
    
    #Build model and fit
    m_f = model_final(x.shape,max_french_sequence_length,english_vocab_size,french_vocab_size)
    m_f.fit(x, y, batch_size=1024, epochs=20, validation_split=0.2)
    
    
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    # Sample sentences
    sentence = 'he saw a old yellow truck'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
    predictions = m_f.predict(sentences, len(sentences))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Il a vu un vieux camion jaune')
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))


final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer)


# ## Optional Enhancements
# 
# In this project we focus on learning/use various network architectures for machine translation. But we could see that the program was able to learn very simple translations. As a next step I gone implement also the very important Attention Models (like the additive attention model and scale the data set.)
