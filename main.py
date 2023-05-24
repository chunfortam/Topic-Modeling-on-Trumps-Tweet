from GPTopic import GPTopic
import numpy as np
import pandas as pd
from Trainer import Trainer
import gensim
import time
from sklearn.cluster import KMeans

def main():
    filename = "bert094-embedding.npy"
    print("Loading data")
    # gptopic = GPTopic(params)
    embeddings = np.load(filename)
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
                      model_name="bert094",
                      params=params,
                      bt_embeddings=embeddings,
                      custom_dataset=True,
                      verbose=True)
        results = trainer.train(save=f"BERTopic_bert094_{i+1}_{time.time()}")
        print(results)
def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

if __name__ == "__main__":
    main()
