from evaluation import DataLoader
import pandas as pd
import os
import openai
import pandas as pd
import tiktoken
import re
import numpy as np
from openai.embeddings_utils import get_embedding
def process_apply(x):
    import openai
    from openai.embeddings_utils import get_embedding
    embedding_model = "text-embedding-ada-002"
    openai.organization = ""
    openai.api_key = "<insert your key>"
    openai.organization = ""
    return get_embedding(x, engine=embedding_model)

def main():
    dataloader = DataLoader(dataset="trump").prepare_docs(save="trump.txt").preprocess_octis(output_folder="trump_414")
    dataset, custom = "trump_414", True
    data_loader = DataLoader(dataset)
    _, timestamps = data_loader.load_docs()
    data = data_loader.load_octis(custom)
    data = [" ".join(words) for words in data.get_corpus()]
    df = pd.DataFrame (data, columns = ['tweets'])
    df = df.dropna()
    df.head(2)

    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # this is the encoding for text-embedding-ada-002
    #max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
    encoding = tiktoken.get_encoding(embedding_encoding)
    df["n_tokens"] = df.tweets.apply(lambda x: len(encoding.encode(x)))
    from pandarallel import pandarallel
    pandarallel.initialize()

    import time
    split_dfs = np.array_split(df,47)
    split_dfs[-1]
    for i in range(len(split_dfs)):
        sdf = split_dfs[i]
        start_time = time.time()
        print("----subset %s" %i)
        print("---- starting %s" %(start_time))
        sdf["embedding"] = sdf["tweets"].parallel_apply(process_apply)
        print("--- %s seconds ---" % (time.time() - start_time))
        sdf["embedding"].head(2)
    resultdf = pd.concat(split_dfs)
    resultdf.head(10)
    resultdf.to_csv("trump_tweet_embedded_v2.csv")
    import ast
    df = pd.read_csv("trump_tweet_embedded.csv")
    emb = np.array([ast.literal_eval(x) for x in df["embedding"]])
    print("emb.shape : %s", str(emb.shape))
    np.save("trump_twitter_np_ada.npy", emb)
    #createStopwordBBERTemb()

def createStopwordBBERTemb():
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic import BERTopic
    df = pd.read_csv("trump_414\\corpus.tsv", names=["tweets"])
    data = df.tweets.tolist()
    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    bert_stoword_emb = topic_model.encode(data, show_progress_bar=True)
    np.save("trump_twitter_bert_stopword_vectorizer_np.npy", bert_stoword_emb)
if __name__ == "__main__":
    main()