Topic Modeling on Trump's Tweets
[Presentation slides ]([url](https://docs.google.com/presentation/d/1WZBpx9AVgXbwttCZD434Ko0MffpT_9zf/edit?usp=sharing&ouid=108707614263026923326&rtpof=true&sd=true))

📌 Project Overview

The goal of this project is to apply topic modeling to Trump's tweets dataset, leveraging state-of-the-art embedding models to improve the quality and coherence of discovered topics. The project includes:

Preprocessing: Cleaning and tokenizing tweets
- Embedding Generation: Comparing Ada-02 (OpenAI) vs. SBERT (Sentence-BERT)
- Dimensionality Reduction: Using UMAP for effective clustering
- Clustering: Leveraging HDBSCAN for discovering tweet groupings
- Topic Representation: Utilizing c-TF-IDF for coherent topic labeling


🔗 Dataset
The dataset consists of 44,253 tweets from Donald Trump.
It includes tweets from different periods, covering political and social events.

⚙️ Setup & Installation
To run this project, ensure you have Python installed. Clone this repository and install dependencies:

```
git clone https://github.com/chunfortam/Topic-Modeling-on-Trumps-Tweet.git  
cd Topic-Modeling-on-Trumps-Tweet  
pip install -r requirements.txt
```

🛠 Required Dependencies

This project requires:
```
openai==0.27.3
octis==1.10.2
bertopic==0.14.1
```
⚠ If you encounter dependency conflicts with octis and other packages, install the latest version of octis using the -e flag to work around the issue:
```
pip install -e git+https://github.com/MIND-Lab/octis.git#egg=octis
```
🔍 Usage
1. Setting Up OpenAI API Key
Before running the script, replace the openai.api_key value in create_embedding.py with your OpenAI API key.

2. Running create_embedding.py
This script processes the tweet dataset and generates embeddings.

```
python src/create_embedding.py
```
After execution, it will output:

- trump.txt: The cleaned text file.
- trump/ (folder): Contains corpus.tsv.
- trump_twitter_np_ada.npy: The generated embeddings.
⚠ Note: If you're using a free OpenAI account, this step may take a long time due to rate limits.

3. Updating File Paths in main.py
If you save the embedding file in a different location, update the variable filename in main.py to reflect the correct path:

```
filename = "trump_twitter_np_ada.npy"
```
Then, execute main.py:

```
python src/main.py
```
This script will:

- Display results for multiple runs with corresponding Topic Coherence (TC) and Topic Diversity (TD) scores.
- Save all results in JSON format under the results/ directory.


4. Calculating Average Performance
To compute the average performance across multiple runs, use sum_field.py.
Modify the file_pattern variable if necessary and run:

```
python src/sum_field.py
```
5. Visualizing Embeddings & Clusters
To visualize vectors after UMAP and HDBSCAN clustering, open and run Visualization.ipynb inside the visual_notebook directory.

📈 Methods
1. Embedding Models
SBERT (Sentence-BERT): A transformer-based embedding model optimized for sentence-level similarity.
Ada-02 (OpenAI): A powerful embedding model used for text similarity and clustering.
2. Dimensionality Reduction & Clustering
UMAP: Reduces high-dimensional embeddings into a lower-dimensional space for better clustering.
HDBSCAN: A density-based clustering algorithm that groups tweets into meaningful topics.
3. Topic Representation
c-TF-IDF: An improved term frequency-inverse document frequency method for extracting meaningful topic words.
🛠️ Tools & Technologies
Python
BERTopic
OpenAI’s Ada-02 Embeddings
SBERT (Sentence-BERT)
UMAP & HDBSCAN
Scikit-learn, Pandas, NumPy
Matplotlib, Seaborn

📜 License
This project is licensed under the MIT License.

