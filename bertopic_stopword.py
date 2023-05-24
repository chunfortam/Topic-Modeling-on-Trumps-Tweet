import numpy as np
import pandas as pd
from Trainer import Trainer
import gensim
import time
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

def before_embedding():
    embeddings = np.load('trump_twitter_bert_stopword_np.npy')
    dataset = "trump"
    print("Finish loading data")

    for i in range(3):
        params = {
        "embedding_model": "BERT",
        "nr_topics": [(i+1)*10 for i in range(5)],
        "min_topic_size": 15,
        #"hdbscan_model":KMeans(n_clusters=50),
        "verbose": True
    }
        print("Training %s set", str(i))

        trainer = Trainer(dataset=dataset,
                      model_name="BERTopic",
                      params=params,
                      bt_embeddings=embeddings,
                      custom_dataset=True,
                      verbose=True)
        results = trainer.train(save=f"BERTopic_trump_stopword_{i+1}_{time.time()}")
        print(results)

before_embedding()

