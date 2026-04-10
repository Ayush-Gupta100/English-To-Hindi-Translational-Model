# 🌐 English to Hindi Neural Machine Translation (Seq2Seq)

## 🚀 Overview

This project implements a **Neural Machine Translation (NMT)** system to translate English sentences into Hindi using a **Sequence-to-Sequence (Seq2Seq)** architecture built with **TensorFlow/Keras**.

It uses:

* Encoder-Decoder architecture
* GRU (Gated Recurrent Units)
* Teacher Forcing for training

---

## 🧠 Model Architecture

### 🔹 Encoder

* Embedding Layer (256 dimensions)
* GRU Layer (256 units)
* Outputs final hidden state

### 🔹 Decoder

* Embedding Layer (256 dimensions)
* GRU Layer (256 units)
* Initialized with encoder hidden state
* Dense layer with softmax activation for prediction

---

## 📊 Data Pipeline

### Input Preparation

| Component        | Shape         | Description                              |
| ---------------- | ------------- | ---------------------------------------- |
| `eng_pad`        | (N, T_enc)    | Tokenized & padded English sentences     |
| `decoder_input`  | (N, T_dec)    | Hindi input shifted left (`<start> ...`) |
| `decoder_target` | (N, T_dec, 1) | Hindi output shifted right (`... <end>`) |

---

## 🔄 Training Strategy

* **Loss Function**: Sparse Categorical Crossentropy
* **Optimizer**: Adam
* **Batch Size**: 16
* **Epochs**: 5
* **Technique**: Teacher Forcing

---

## 🏗️ Model Implementation

```python
encoder_input = Input(shape=(None,))
encoder_emb = Embedding(vocab1, 256)(encoder_input)

encoder_output, encoder_state = GRU(
    256, return_sequences=True, return_state=True
)(encoder_emb)

decoder_input = Input(shape=(None,))
decoder_emb = Embedding(vocab2, 256)(decoder_input)

decoder_output, _ = GRU(
    256, return_sequences=True, return_state=True
)(
    decoder_emb, initial_state=encoder_state
)

output = Dense(vocab2, activation='softmax')(decoder_output)

model = Model([encoder_input, decoder_input], output)
```

---

## ▶️ How to Train

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

model.fit(
    [eng_pad, decoder_input_data],
    decoder_target,
    batch_size=16,
    epochs=5,
    validation_split=0.1
)
```

---

## 📌 Key Concepts

### 🔁 Teacher Forcing

During training, the correct previous word is fed into the decoder instead of predicted output.

### 🔀 Sequence Shifting

* Decoder Input → `<start> मैं ठीक हूँ`
* Target Output → `मैं ठीक हूँ <end>`

---

## ⚠️ Common Issues & Fixes

| Issue                 | Solution                                        |
| --------------------- | ----------------------------------------------- |
| GRU state shape error | Ensure `initial_state=encoder_state` (not list) |
| Model not training    | Pass numpy arrays, not Keras tensors            |
| Shape mismatch        | Ensure `(N, T)` and `(N, T, 1)` formats         |
| Embedding errors      | Use `int32` tokenized inputs                    |

---

## 📈 Future Improvements

* 🔥 Add Attention Mechanism
* 🚀 Use Transformer architecture
* 📊 BLEU score evaluation
* 🌍 Deploy as API (Flask/FastAPI)
* 📱 Build UI for real-time translation

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy

---

## 📌 Project Status

✅ Training pipeline complete
🚧 Inference model (next step)
🚀 Ready for optimization

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Ayush Gupta**
