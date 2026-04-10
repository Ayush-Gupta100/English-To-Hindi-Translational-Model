import pickle
import preprocess
import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)]
    )


data=pd.read_csv('Hindi_English_Truncated_Corpus.csv')
print(data.sample(10))
eng_sent,token1,hindi_sent,token2,d_in,d_target=preprocess.preprocess(data)
print(eng_sent)
print("Vocab size eng :",len(token1.word_index)+1)
print("----------------------------")
print("Vocab size Hindi :",len(token2.word_index)+1)

ENG_VOCAB_SIZE = min(20000, len(token1.word_index) + 1)
HIN_VOCAB_SIZE = min(20000, len(token2.word_index) + 1)

encoder_input=tf.keras.Input(shape=(None,))
encoder_embed=Embedding(ENG_VOCAB_SIZE,256,mask_zero=True)(encoder_input)
encoder_gru=GRU(256,return_state=True,return_sequences=True)
encoder_outputs,encoder_state=encoder_gru(encoder_embed)

decoder_input=tf.keras.Input(shape=(None,))
decoder_embed=Embedding(HIN_VOCAB_SIZE,256,mask_zero=True)(decoder_input)
decoder_gru=GRU(256,return_state=True,return_sequences=True)

decoder_outputs,decoder_states=decoder_gru(decoder_embed,initial_state=encoder_state)

decoder_dense=Dense(HIN_VOCAB_SIZE,activation='softmax',dtype="float16")
output=decoder_dense(decoder_outputs)

model=tf.keras.models.Model(
    inputs=[encoder_input,decoder_input],
    outputs=output
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

print(model.summary())
print(d_target)

with open("eng_tokenizer.pkl", "wb") as f:
    pickle.dump(token1, f)
with open("hin_tokenizer.pkl", "wb") as f:
    pickle.dump(token2, f)

tf.keras.utils.plot_model(
    model, 
    to_file='model_architecture.png', 
    show_shapes=True, 
    show_layer_names=True,
    expand_nested=True,
    dpi=96
)

with tf.device('/GPU:0'):
    try:
        history = model.fit(
        [eng_sent,d_in],
        d_target,
        batch_size=16,
        epochs=5,
        validation_split=0.1
        )
        model.save("eng_hin_seq2seq_gru")

    except Exception as e:
        print("Training not completed:\n", e)


