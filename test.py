import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import load_model

model = load_model("eng_hin_seq2seq_gru_attention")

print(model.summary())

text=input("Enter English sentence: ")

with open("eng_tokenizer.pkl", "rb") as f:
    eng_tokenizer=pickle.load(f)
with open("hin_tokenizer.pkl", "rb") as f:
    hin_tokenizer=pickle.load(f)

input_embedding=eng_tokenizer.texts_to_sequences([text])
input_embedding=tf.keras.preprocessing.sequence.pad_sequences(input_embedding,maxlen=50,padding='post')

test_input = np.zeros((1, 49))
test_input[0, 0] = hin_tokenizer.word_index['start'] 
predicted_sentence = []

print([input_embedding, test_input])
for i in range(48):
    output = model.predict([input_embedding, test_input], verbose=0)    
    predicted_id = np.argmax(output[0, i, :])
    if i + 1 < 49:
        test_input[0, i + 1] = predicted_id
    word = hin_tokenizer.index_word.get(predicted_id, "")
    if word == 'end' or word == "":
        break
    predicted_sentence.append(word)
print("Actual Translation:", " ".join(predicted_sentence))


