# 📊 Clustering and Visualizing Text Data for Similarity

## 📝 Overview  
This project explores various techniques for clustering text data based on semantic similarity.  
We experiment with several embedding methods—**TF-IDF**, **GloVe**, **SentenceTransformer (SBERT)**, and **BERT**—and compare clustering performance using **KMeans** and **DBSCAN**.  
The dataset used is **20 Newsgroups** with 3 selected categories.

---

## 🎯 Project Objectives

- **Preprocessing:** Lowercasing, URL/special character removal, stopword removal, and lemmatization  
- **Embedding Techniques:**  
  - TF-IDF  
  - GloVe  
  - SentenceTransformer (SBERT)  
  - BERT  
- **Clustering Algorithms:**  
  - KMeans  
  - DBSCAN  
- **Visualization:**  
  - PCA  
  - t-SNE  
- **Evaluation Metrics:**  
  - Adjusted Rand Index (ARI)  
  - Normalized Mutual Information (NMI)  
  - Fowlkes-Mallows Index (FMI)  
- **Model Saving:** Serialize best model (SBERT + KMeans) for inference

---

## 📁 Project Structure

- `final_sbert_kmeans_text_clustering.ipynb` – Main Jupyter notebook  
- `sbert_kmeans_model.pkl` – Saved best model (KMeans)  
- `sbert_embeddings.npy` – Saved SBERT embeddings (optional reuse)  
- `README.md` – This file

---

## ⚙️ Setup and Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your_username/your_repo_name.git  
cd your_repo_name
```

### 2️⃣ Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

📌 *Note: Make sure `requirements.txt` includes: numpy, pandas, nltk, scikit-learn, sentence-transformers, transformers, torch, matplotlib, seaborn*

### 3️⃣ Run the Notebook  
Open the notebook and execute all cells:
```bash
final_sbert_kmeans_text_clustering.ipynb
```

### 4️⃣ Inference Example

```python
from sentence_transformers import SentenceTransformer  
import joblib  

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  
kmeans_model = joblib.load('sbert_kmeans_model.pkl')  

# Your text input
text = "Your custom text here"
processed = preprocess(text)  # Apply same preprocessing
embedding = sbert_model.encode(processed)
label = kmeans_model.predict([embedding])[0]
print("Predicted Cluster:", label)
```

---

## 📊 Results

**Best Combination:**
- **Embedding:** SentenceTransformer (SBERT)  
- **Clustering:** KMeans  
- **Metrics:**  
  - ARI: `0.8358`  
  - NMI: `0.7934`  
  - FMI: `0.8907`  

✅ SBERT + KMeans showed the most accurate and well-separated clusters.

---

## 🧠 Code Structure and Best Practices

- Functions defined for **preprocessing**, **embedding**, and **evaluation**  
- Clustering logic separated clearly  
- Basic **DRY principles** followed

---

## ✅ Conclusion

This project demonstrates how modern embedding techniques (like SBERT) combined with clustering algorithms (like KMeans) can effectively group and visualize semantically similar documents.

**Applications include:**  
- Market research  
- Customer segmentation  
- Topic discovery
