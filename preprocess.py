import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess(data):
    data.drop(columns=['source'],inplace=True)
    data.dropna(inplace=True)
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: '<start> ' + x + ' <end>')
    tokenizer1 = Tokenizer(num_words=20000, oov_token='<OOV>')
    tokenizer2 = Tokenizer(num_words=20000, oov_token='<OOV>')

    tokenizer1.fit_on_texts(data['english_sentence'].values)
    tokenizer2.fit_on_texts(data['hindi_sentence'].values)
    
    eng_seq=tokenizer1.texts_to_sequences(data['english_sentence'].values)
    hindi_seq=tokenizer2.texts_to_sequences(data['hindi_sentence'].values)
    
    max_len_eng = max([len(i) for i in eng_seq])
    max_len_hin = max([len(i) for i in hindi_seq])

    encoder_input = pad_sequences(eng_seq, maxlen=20, padding='post')
    pad_sequence2 = pad_sequences(hindi_seq, maxlen=20, padding='post')

    decoder_input=pad_sequence2[:,:-1]
    decoder_target=pad_sequence2[:,1:]

    decoder_target=decoder_target[..., np.newaxis]
    
    return encoder_input, tokenizer1, pad_sequence2, tokenizer2, decoder_input, decoder_target
    

