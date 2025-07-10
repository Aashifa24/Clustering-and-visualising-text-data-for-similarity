Clustering and Visualizing Text Data for Similarity
Overview
This project explores various techniques for clustering text data based on semantic similarity. We experiment with several embedding methods—including TF-IDF, GloVe, SentenceTransformer (SBERT), and BERT—and compare clustering performance using algorithms like KMeans and DBSCAN. The goal is to identify the best combination for grouping similar documents using the 20 Newsgroups dataset (with 3 selected categories).

Project Objectives
Preprocessing: Apply standard NLP preprocessing steps (lowercasing, URL/special character removal, stopword removal, and lemmatization).

Embedding Techniques: Explore multiple text representations:

TF-IDF

GloVe

SentenceTransformer (SBERT)

BERT

Clustering Algorithms: Evaluate two clustering methods:

KMeans

DBSCAN

Visualization: Use PCA (and t-SNE) to visualize clustering results in 2D.

Evaluation: Measure clustering quality using Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and Fowlkes-Mallows Index (FMI).

Model Saving: Serialize the best model (SBERT + KMeans) for future inference.

Project Structure
Notebook: The main analysis is done in the Jupyter Notebook (e.g., final_sbert_kmeans_text_clustering.ipynb).

Model Serialization: The final best model is saved as sbert_kmeans_model.pkl, with optional embeddings saved as sbert_embeddings.npy.

README.md: This file summarizes the project, its findings, and usage instructions.

Setup and Usage
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/your_username/your_repo_name.git  
cd your_repo_name  
Create a Virtual Environment and Install Dependencies:

bash
Copy
Edit
python -m venv venv  
# On Windows:  
venv\Scripts\activate  
# On macOS/Linux:  
source venv/bin/activate  

pip install -r requirements.txt  
Note: The requirements.txt should list libraries like numpy, pandas, nltk, scikit-learn, sentence-transformers, transformers, torch, matplotlib, seaborn.

Run the Notebook:
Open final_sbert_kmeans_text_clustering.ipynb and execute the cells step-by-step to reproduce the results.

Inference:
After training, load the SBERT + KMeans model for inference:

python
Copy
Edit
from sentence_transformers import SentenceTransformer  
import joblib  

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  
kmeans_model = joblib.load('sbert_kmeans_model.pkl')  

# Preprocess input text  
text = "Your custom text here"  
processed_text = preprocess(text)  # Use your preprocessing function  
embedding = sbert_model.encode(processed_text)  
predicted_cluster = kmeans_model.predict([embedding])[0]  
print("Predicted Cluster:", predicted_cluster)  
Results
After evaluating all combinations, the best performing setup was:

Embedding: SentenceTransformer (SBERT)

Clustering: KMeans

Metrics: ARI: 0.8358, NMI: 0.7934, FMI: 0.8907

This confirms that SBERT + KMeans is the most effective approach for clustering text similarity in this project.

Code Organization and Best Practices
The code follows basic DRY and modular practices.

Functions are defined for preprocessing, embedding generation, and evaluation.

Clustering and visualization steps are clearly segmented for readability.

Conclusion
This project demonstrates effective techniques for clustering and visualizing text data. The successful combination of SBERT embeddings with KMeans clustering yields accurate and interpretable clusters, which can be further applied in market research, segmentation, sentiment analysis, and more.
