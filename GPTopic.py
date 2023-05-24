##The following class is a modified version of BERTopic to be used for GPT-4/Ada02
##https://github.com/MaartenGr/BERTopic

import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    yaml._warnings_enabled["YAMLLoadWarning"] = False
except (KeyError, AttributeError, TypeError) as e:
    pass
import logging
import re
import math
import joblib
import inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from packaging import version
from scipy.sparse import csr_matrix
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable

# Models
import hdbscan
from umap import UMAP
from sklearn.preprocessing import normalize
from sklearn import __version__ as sklearn_version
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# BERTopic
from bertopic import plotting
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.backend import BaseEmbedder
from bertopic.backend._utils import select_backend
from bertopic.representation import BaseRepresentation
from bertopic.cluster._utils import hdbscan_delegator, is_supported_hdbscan
from bertopic._utils import MyLogger, check_documents_type, check_embeddings_shape, check_is_fitted


logger = MyLogger("WARNING")
class GPTopic:
    """

Attributes:
    topics_ (List[int]) : The topics that are generated for each document after training or updating
                          the topic model. The most recent topics are tracked.
    probabilities_ (List[float]): The probability of the assigned topic per document. These are
                                  only calculated if a HDBSCAN model is used for the clustering step.
                                  When `calculate_probabilities=True`, then it is the probabilities
                                  of all topics per document.
    topic_sizes_ (Mapping[int, int]) : The size of each topic
    topic_mapper_ (TopicMapper) : A class for tracking topics and their mappings anytime they are
                                  merged, reduced, added, or removed.
    topic_representations_ (Mapping[int, Tuple[int, float]]) : The top n terms per topic and their respective
                                                               c-TF-IDF values.
    c_tf_idf_ (csr_matrix) : The topic-term matrix as calculated through c-TF-IDF. To access its respective
                             words, run `.vectorizer_model.get_feature_names()`  or
                             `.vectorizer_model.get_feature_names_out()`
    topic_labels_ (Mapping[int, str]) : The default labels for each topic.
    custom_labels_ (List[str]) : Custom labels for each topic.
    topic_embeddings_ (np.ndarray) : The embeddings for each topic. It is calculated by taking the
                                     weighted average of word embeddings in a topic based on their c-TF-IDF values.
    representative_docs_ (Mapping[int, str]) : The representative documents for each topic.

Examples:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all')['data']
topic_model = BERTopic()
topics, probabilities = topic_model.fit_transform(docs)
```

If you want to use your own embedding model, use it as follows:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

docs = fetch_20newsgroups(subset='all')['data']
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=sentence_model)
```

Due to the stochastisch nature of UMAP, the results from BERTopic might differ
and the quality can degrade. Using your own embeddings allows you to
try out BERTopic several times until you find the topics that suit
you best.
"""
    def __init__(self,
                 language: str = "english",
                 top_n_words: int = 10,
                 n_gram_range: Tuple[int, int] = (1, 1),
                 min_topic_size: int = 10,
                 nr_topics: Union[int, str] = None,
                 low_memory: bool = False,
                 calculate_probabilities: bool = False,
                 seed_topic_list: List[List[str]] = None,
                 embedding_model=None,
                 umap_model: UMAP = None,
                 hdbscan_model: hdbscan.HDBSCAN = None,
                 vectorizer_model: CountVectorizer = None,
                 ctfidf_model: TfidfTransformer = None,
                 representation_model: BaseRepresentation = None,
                 verbose: bool = False,
                 ):
        """initialization

        Arguments:
            language: The main language used in your documents. The default sentence-transformers
                      model for "english" is `all-MiniLM-L6-v2`. For a full overview of
                      supported languages see bertopic.backend.languages. Select
                      "multilingual" to load in the `paraphrase-multilingual-MiniLM-L12-v2`
                      sentence-tranformers model that supports 50+ languages.
                      NOTE: This is not used if `embedding_model` is used.
            top_n_words: The number of words per topic to extract. Setting this
                         too high can negatively impact topic embeddings as topics
                         are typically best represented by at most 10 words.
            n_gram_range: The n-gram range for the CountVectorizer.
                          Advised to keep high values between 1 and 3.
                          More would likely lead to memory issues.
                          NOTE: This param will not be used if you pass in your own
                          CountVectorizer.
            min_topic_size: The minimum size of the topic. Increasing this value will lead
                            to a lower number of clusters/topics.
                            NOTE: This param will not be used if you are not using HDBSCAN.
            nr_topics: Specifying the number of topics will reduce the initial
                       number of topics to the value specified. This reduction can take
                       a while as each reduction in topics (-1) activates a c-TF-IDF
                       calculation. If this is set to None, no reduction is applied. Use
                       "auto" to automatically reduce topics using HDBSCAN.
            low_memory: Sets UMAP low memory to True to make sure less memory is used.
                        NOTE: This is only used in UMAP. For example, if you use PCA instead of UMAP
                        this parameter will not be used.
            calculate_probabilities: Calculate the probabilities of all topics
                                     per document instead of the probability of the assigned
                                     topic per document. This could slow down the extraction
                                     of topics if you have many documents (> 100_000).
                                     NOTE: If false you cannot use the corresponding
                                     visualization method `visualize_probabilities`.
                                     NOTE: This is an approximation of topic probabilities
                                     as used in HDBSCAN and not an exact representation.
            seed_topic_list: A list of seed words per topic to converge around
            verbose: Changes the verbosity of the model, Set to True if you want
                     to track the stages of the model.
            embedding_model: Use a custom embedding model.
                             The following backends are currently supported
                               * SentenceTransformers
                               * Flair
                               * Spacy
                               * Gensim
                               * USE (TF-Hub)
                             You can also pass in a string that points to one of the following
                             sentence-transformers models:
                               * https://www.sbert.net/docs/pretrained_models.html
            umap_model: Pass in a UMAP model to be used instead of the default.
                        NOTE: You can also pass in any dimensionality reduction algorithm as long
                        as it has `.fit` and `.transform` functions.
            hdbscan_model: Pass in a hdbscan.HDBSCAN model to be used instead of the default
                           NOTE: You can also pass in any clustering algorithm as long as it has
                           `.fit` and `.predict` functions along with the `.labels_` variable.
            vectorizer_model: Pass in a custom `CountVectorizer` instead of the default model.
            ctfidf_model: Pass in a custom ClassTfidfTransformer instead of the default model.
            representation_model: Pass in a model that fine-tunes the topic representations
                                  calculated through c-TF-IDF. Models from `bertopic.representation`
                                  are supported.
        """
        # Topic-based parameters
        if top_n_words > 100:
            warnings.warn("Note that extracting more than 100 words from a sparse "
                          "can slow down computation quite a bit.")

        self.top_n_words = top_n_words
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.low_memory = low_memory
        self.calculate_probabilities = calculate_probabilities
        self.verbose = verbose
        self.seed_topic_list = seed_topic_list

        # Embedding model
        self.language = language if not embedding_model else None
        self.embedding_model = embedding_model

        # Vectorizer
        self.n_gram_range = n_gram_range
        self.vectorizer_model = vectorizer_model or CountVectorizer(ngram_range=self.n_gram_range)
        self.ctfidf_model = ctfidf_model or ClassTfidfTransformer()

        # Representation model
        self.representation_model = representation_model

        # UMAP or another algorithm that has .fit and .transform functions
        self.umap_model = umap_model or UMAP(n_neighbors=15,
                                             n_components=5,
                                             min_dist=0.0,
                                             metric='cosine',
                                             low_memory=self.low_memory)

        # HDBSCAN or another clustering algorithm that has .fit and .predict functions and
        # the .labels_ variable to extract the labels
        if not hdbscan_model:
            print("creating new HDBSCAN model")
        else:
            print("Using Kmean")
        self.hdbscan_model = hdbscan_model or hdbscan.HDBSCAN(min_cluster_size=self.min_topic_size,
                                                              metric='euclidean',
                                                              cluster_selection_method='eom',
                                                              prediction_data=True)

        # Public attributes
        self.topics_ = None
        self.probabilities_ = None
        self.topic_sizes_ = None
        self.topic_mapper_ = None
        self.topic_representations_ = None
        self.topic_embeddings_ = None
        self.topic_labels_ = None
        self.custom_labels_ = None
        self.representative_docs_ = {}
        self.c_tf_idf_ = None

        # Private attributes for internal tracking purposes
        self._outliers = 1
        self._merged_topics = None
    def fit_transform(self,
                      documents: List[str],
                      embeddings: np.ndarray = None,
                      y: Union[List[int], np.ndarray] = None) -> Tuple[List[int],
                                                                       Union[np.ndarray, None]]:
        """ Fit the models on a collection of documents, generate topics, and return the docs with topics

        Arguments:
            documents: A list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model
            y: The target class for (semi)-supervised modeling. Use -1 if no class for a
               specific instance is specified.

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The probability of the assigned topic per document.
                           If `calculate_probabilities` in BERTopic is set to True, then
                           it calculates the probabilities of all topics across all documents
                           instead of only the assigned topic. This, however, slows down
                           computation and may increase memory usage.

        Examples:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        docs = fetch_20newsgroups(subset='all')['data']
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        ```

        If you want to use your own embeddings, use it as follows:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Create topic model
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs, embeddings)
        ```
        """
        check_documents_type(documents)
        check_embeddings_shape(embeddings, documents)

        documents = pd.DataFrame({"Document": documents,
                                  "ID": range(len(documents)),
                                  "Topic": None})

        # Extract embeddings
        # if embeddings is None:
        #     self.embedding_model = select_backend(self.embedding_model,
        #                                           language=self.language)
        #     embeddings = self._extract_embeddings(documents.Document,
        #                                           method="document",
        #                                           verbose=self.verbose)
        #     logger.info("Transformed documents to Embeddings")
        # else:
        #     if self.embedding_model is not None:
        #         self.embedding_model = select_backend(self.embedding_model,
        #                                               language=self.language)
        self.embedding_model = "GPT"

        # Reduce dimensionality
        # if self.seed_topic_list is not None and self.embedding_model is not None:
        #     y, embeddings = self._guided_topic_modeling(embeddings)
        umap_embeddings = self._reduce_dimensionality(embeddings, y)

        # Cluster reduced embeddings
        documents, probabilities = self._cluster_embeddings(umap_embeddings, documents, y=y)

        # Sort and Map Topic IDs by their frequency
        if not self.nr_topics:
            documents = self._sort_mappings_by_frequency(documents)

        # Extract topics by calculating c-TF-IDF
        self._extract_topics(documents)

        # Reduce topics
        if self.nr_topics:
            documents = self._reduce_topics(documents)

        # Save the top 3 most representative documents per topic
        self._save_representative_docs(documents)

        # Resulting output
        self.probabilities_ = self._map_probabilities(probabilities, original_topics=True)
        predictions = documents.Topic.to_list()

        return predictions, self.probabilities_
    def _map_probabilities(self,
                           probabilities: Union[np.ndarray, None],
                           original_topics: bool = False) -> Union[np.ndarray, None]:
        """ Map the probabilities to the reduced topics.
        This is achieved by adding the probabilities together
        of all topics that were mapped to the same topic. Then,
        the topics that were mapped from were set to 0 as they
        were reduced.

        Arguments:
            probabilities: An array containing probabilities
            original_topics: Whether we want to map from the
                             original topics to the most recent topics
                             or from the second-most recent topics.

        Returns:
            mapped_probabilities: Updated probabilities
        """
        mappings = self.topic_mapper_.get_mappings(original_topics)

        # Map array of probabilities (probability for assigned topic per document)
        if probabilities is not None:
            if len(probabilities.shape) == 2:
                mapped_probabilities = np.zeros((probabilities.shape[0],
                                                 len(set(mappings.values())) - self._outliers))
                for from_topic, to_topic in mappings.items():
                    if to_topic != -1 and from_topic != -1:
                        mapped_probabilities[:, to_topic] += probabilities[:, from_topic]

                return mapped_probabilities

        return probabilities
    def _save_representative_docs(self, documents: pd.DataFrame):
        """ Save the 3 most representative docs per topic

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

        Updates:
            self.representative_docs_: Populate each topic with 3 representative docs
        """
        repr_docs, _, _= self._extract_representative_docs(self.c_tf_idf_,
                                                           documents,
                                                           self.topic_representations_,
                                                           nr_samples=500,
                                                           nr_repr_docs=3)
        self.representative_docs_ = repr_docs
    def _extract_representative_docs(self,
                                     c_tf_idf: csr_matrix,
                                     documents: pd.DataFrame,
                                     topics: Mapping[str, List[Tuple[str, float]]],
                                     nr_samples: int = 500,
                                     nr_repr_docs: int = 5,
                                     ) -> Union[List[str], List[List[int]]]:
        """ Approximate most representative documents per topic by sampling
        a subset of the documents in each topic and calculating which are
        most represenative to their topic based on the cosine similarity between
        c-TF-IDF representations.

        Arguments:
            c_tf_idf: The topic c-TF-IDF representation
            documents: All input documents
            topics: The candidate topics as calculated with c-TF-IDF
            nr_samples: The number of candidate documents to extract per topic
            nr_repr_docs: The number of representative documents to extract per topic

        Returns:
            repr_docs_mappings: A dictionary from topic to representative documents
            representative_docs: A flat list of representative documents
            repr_doc_indices: The indices of representative documents
                              that belong to each topic
        """
        # Sample documents per topic
        documents_per_topic = (
            documents.groupby('Topic')
                     .sample(n=nr_samples, replace=True, random_state=42)
                     .drop_duplicates()
        )

        # Find and extract documents that are most similar to the topic
        repr_docs = []
        repr_docs_indices = []
        repr_docs_mappings = {}
        labels = sorted(list(topics.keys()))
        for index, topic in enumerate(labels):

            # Calculate similarity
            selected_docs = documents_per_topic.loc[documents_per_topic.Topic == topic, "Document"].values
            bow = self.vectorizer_model.transform(selected_docs)
            ctfidf = self.ctfidf_model.transform(bow)
            sim_matrix = cosine_similarity(ctfidf, c_tf_idf[index])

            # Extract top n most representative documents
            nr_docs = nr_repr_docs if len(selected_docs) > nr_repr_docs else len(selected_docs)
            indices = np.argpartition(sim_matrix.reshape(1, -1)[0],
                                      -nr_docs)[-nr_docs:]
            repr_docs.extend([selected_docs[index] for index in indices])
            repr_docs_indices.append([repr_docs_indices[-1][-1] + i + 1 if index != 0 else i for i in range(nr_docs)])
        repr_docs_mappings = {topic: repr_docs[i[0]:i[-1]+1] for topic, i in zip(topics.keys(), repr_docs_indices)}

        return repr_docs_mappings, repr_docs, repr_docs_indices
    def get_topics(self) -> Mapping[str, Tuple[str, float]]:
        """ Return topics with top n words and their c-TF-IDF score

        Returns:
            self.topic_representations_: The top n words per topic and the corresponding c-TF-IDF score

        Examples:

        ```python
        all_topics = topic_model.get_topics()
        ```
        """
        check_is_fitted(self)
        return self.topic_representations_
    def _reduce_to_n_topics(self, documents: pd.DataFrame) -> pd.DataFrame:
        """ Reduce topics to self.nr_topics

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        topics = documents.Topic.tolist().copy()

        # Create topic distance matrix
        if self.topic_embeddings_ is not None:
            topic_embeddings = np.array(self.topic_embeddings_)[self._outliers:, ]
        else:
            topic_embeddings = self.c_tf_idf_[self._outliers:, ].toarray()
        distance_matrix = 1-cosine_similarity(topic_embeddings)
        np.fill_diagonal(distance_matrix, 0)

        # Cluster the topic embeddings using AgglomerativeClustering
        if version.parse(sklearn_version) >= version.parse("1.4.0"):
            cluster = AgglomerativeClustering(self.nr_topics - self._outliers, metric="precomputed", linkage="average")
        else:
            cluster = AgglomerativeClustering(self.nr_topics - self._outliers, affinity="precomputed", linkage="average")
        cluster.fit(distance_matrix)
        new_topics = [cluster.labels_[topic] if topic != -1 else -1 for topic in topics]

        # Map topics
        documents.Topic = new_topics
        self._update_topic_size(documents)
        mapped_topics = {from_topic: to_topic for from_topic, to_topic in zip(topics, new_topics)}
        self.topic_mapper_.add_mappings(mapped_topics)

        # Update representations
        documents = self._sort_mappings_by_frequency(documents)
        self._extract_topics(documents)
        self._update_topic_size(documents)
        return documents
    def _reduce_topics(self, documents: pd.DataFrame) -> pd.DataFrame:
        """ Reduce topics to self.nr_topics

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        initial_nr_topics = len(self.get_topics())

        if isinstance(self.nr_topics, int):
            if self.nr_topics < initial_nr_topics:
                documents = self._reduce_to_n_topics(documents)
        elif isinstance(self.nr_topics, str):
            documents = self._auto_reduce_topics(documents)
        else:
            raise ValueError("nr_topics needs to be an int or 'auto'! ")

        logger.info(f"Reduced number of topics from {initial_nr_topics} to {len(self.get_topic_freq())}")
        return documents
    def get_topic_freq(self, topic: int = None) -> Union[pd.DataFrame, int]:
        """ Return the the size of topics (descending order)

        Arguments:
            topic: A specific topic for which you want the frequency

        Returns:
            Either the frequency of a single topic or dataframe with
            the frequencies of all topics

        Examples:

        To extract the frequency of all topics:

        ```python
        frequency = topic_model.get_topic_freq()
        ```

        To get the frequency of a single topic:

        ```python
        frequency = topic_model.get_topic_freq(12)
        ```
        """
        check_is_fitted(self)
        if isinstance(topic, int):
            return self.topic_sizes_[topic]
        else:
            return pd.DataFrame(self.topic_sizes_.items(), columns=['Topic', 'Count']).sort_values("Count",
                                                                                                   ascending=False)

    def _auto_reduce_topics(self, documents: pd.DataFrame) -> pd.DataFrame:
        """ Reduce the number of topics automatically using HDBSCAN

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        topics = documents.Topic.tolist().copy()
        unique_topics = sorted(list(documents.Topic.unique()))[self._outliers:]
        max_topic = unique_topics[-1]

        # Find similar topics
        if self.topic_embeddings_ is not None:
            embeddings = np.array(self.topic_embeddings_)
        else:
            embeddings = self.c_tf_idf_.toarray()
        norm_data = normalize(embeddings, norm='l2')
        predictions = hdbscan.HDBSCAN(min_cluster_size=2,
                                      metric='euclidean',
                                      cluster_selection_method='eom',
                                      prediction_data=True).fit_predict(norm_data[self._outliers:])

        # Map similar topics
        mapped_topics = {unique_topics[index]: prediction + max_topic
                         for index, prediction in enumerate(predictions)
                         if prediction != -1}
        documents.Topic = documents.Topic.map(mapped_topics).fillna(documents.Topic).astype(int)
        mapped_topics = {from_topic: to_topic for from_topic, to_topic in zip(topics, documents.Topic.tolist())}

        # Update documents and topics
        self.topic_mapper_.add_mappings(mapped_topics)
        documents = self._sort_mappings_by_frequency(documents)
        self._extract_topics(documents)
        self._update_topic_size(documents)
        return documents

    def _extract_embeddings(self,
                            documents: Union[List[str], str],
                            method: str = "document",
                            verbose: bool = None) -> np.ndarray:
        """ Extract sentence/document embeddings through pre-trained embeddings
        For an overview of pre-trained models: https://www.sbert.net/docs/pretrained_models.html

        Arguments:
            documents: Dataframe with documents and their corresponding IDs
            method: Whether to extract document or word-embeddings, options are "document" and "word"
            verbose: Whether to show a progressbar demonstrating the time to extract embeddings

        Returns:
            embeddings: The extracted embeddings.
        """
        return np.load('trump_twitter_np_ada.npy')
    def _reduce_dimensionality(self,
                               embeddings: Union[np.ndarray, csr_matrix],
                               y: Union[List[int], np.ndarray] = None,
                               partial_fit: bool = False) -> np.ndarray:
        """ Reduce dimensionality of embeddings using UMAP and train a UMAP model

        Arguments:
            embeddings: The extracted embeddings using the sentence transformer module.
            y: The target class for (semi)-supervised dimensionality reduction
            partial_fit: Whether to run `partial_fit` for online learning

        Returns:
            umap_embeddings: The reduced embeddings
        """
        # Partial fit
        if partial_fit:
            if hasattr(self.umap_model, "partial_fit"):
                self.umap_model = self.umap_model.partial_fit(embeddings)
            elif self.topic_representations_ is None:
                self.umap_model.fit(embeddings)

        # Regular fit
        else:
            try:
                self.umap_model.fit(embeddings, y=y)
            except TypeError:
                logger.info("The dimensionality reduction algorithm did not contain the `y` parameter and"
                            " therefore the `y` parameter was not used")
                self.umap_model.fit(embeddings)

        umap_embeddings = self.umap_model.transform(embeddings)
        logger.info("Reduced dimensionality")
        return np.nan_to_num(umap_embeddings)
    def _sort_mappings_by_frequency(self, documents: pd.DataFrame) -> pd.DataFrame:
        """ Reorder mappings by their frequency.

        For example, if topic 88 was mapped to topic
        5 and topic 5 turns out to be the largest topic,
        then topic 5 will be topic 0. The second largest,
        will be topic 1, etc.

        If there are no mappings since no reduction of topics
        took place, then the topics will simply be ordered
        by their frequency and will get the topic ids based
        on that order.

        This means that -1 will remain the outlier class, and
        that the rest of the topics will be in descending order
        of ids and frequency.

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics

        Returns:
            documents: Updated dataframe with documents and the mapped
                       and re-ordered topic ids
        """
        self._update_topic_size(documents)

        # Map topics based on frequency
        df = pd.DataFrame(self.topic_sizes_.items(), columns=["Old_Topic", "Size"]).sort_values("Size", ascending=False)
        df = df[df.Old_Topic != -1]
        sorted_topics = {**{-1: -1}, **dict(zip(df.Old_Topic, range(len(df))))}
        self.topic_mapper_.add_mappings(sorted_topics)

        # Map documents
        documents.Topic = documents.Topic.map(sorted_topics).fillna(documents.Topic).astype(int)
        self._update_topic_size(documents)
        return documents
    def _update_topic_size(self, documents: pd.DataFrame):
        """ Calculate the topic sizes

        Arguments:
            documents: Updated dataframe with documents and their corresponding IDs and newly added Topics
        """
        sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
        self.topic_sizes_ = dict(zip(sizes.Topic, sizes.Document))
        self.topics_ = documents.Topic.astype(int).tolist()
    def _extract_topics(self, documents: pd.DataFrame):
        """ Extract topics from the clusters using a class-based TF-IDF

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

        Returns:
            c_tf_idf: The resulting matrix giving a value (importance score) for each word per topic
        """
        ##grouping the tweets by hdbscan label
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        ##this is calculating the ctf_idf for each word, and the words themselves
        self.c_tf_idf_, words = self._c_tf_idf(documents_per_topic)
        self.topic_representations_ = self._extract_words_per_topic(words, documents)
        self._create_topic_vectors()
        self.topic_labels_ = {key: f"{key}_" + "_".join([word[0] for word in values[:4]])
                              for key, values in
                              self.topic_representations_.items()}
    def _preprocess_text(self, documents: np.ndarray) -> List[str]:
        """ Basic preprocessing of text

        Steps:
            * Replace \n and \t with whitespace
            * Only keep alpha-numerical characters
        """
        cleaned_documents = [doc.replace("\n", " ") for doc in documents]
        cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
        if self.language == "english":
            cleaned_documents = [re.sub(r'[^A-Za-z0-9 ]+', '', doc) for doc in cleaned_documents]
        cleaned_documents = [doc if doc != "" else "emptydoc" for doc in cleaned_documents]
        return cleaned_documents
    def _c_tf_idf(self,
                  documents_per_topic: pd.DataFrame,
                  fit: bool = True,
                  partial_fit: bool = False) -> Tuple[csr_matrix, List[str]]:
        """ Calculate a class-based TF-IDF where m is the number of total documents.

        Arguments:
            documents_per_topic: The joined documents per topic such that each topic has a single
                                 string made out of multiple documents
            m: The total number of documents (unjoined)
            fit: Whether to fit a new vectorizer or use the fitted self.vectorizer_model
            partial_fit: Whether to run `partial_fit` for online learning

        Returns:
            tf_idf: The resulting matrix giving a value (importance score) for each word per topic
            words: The names of the words to which values were given
        """
        documents = self._preprocess_text(documents_per_topic.Document.values)

        if partial_fit:
            X = self.vectorizer_model.partial_fit(documents).update_bow(documents)
        elif fit:
            self.vectorizer_model.fit(documents)
            X = self.vectorizer_model.transform(documents)
        else:
            X = self.vectorizer_model.transform(documents)

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            ##this is just extracting words out of the document
            words = self.vectorizer_model.get_feature_names_out()
        else:
            words = self.vectorizer_model.get_feature_names()

        if self.seed_topic_list:
            seed_topic_list = [seed for seeds in self.seed_topic_list for seed in seeds]
            multiplier = np.array([1.2 if word in seed_topic_list else 1 for word in words])
        else:
            multiplier = None

        if fit:
            self.ctfidf_model = self.ctfidf_model.fit(X, multiplier=multiplier)
        ##ctifidf fit just calculate the score
        c_tf_idf = self.ctfidf_model.transform(X)

        return c_tf_idf, words
    def _cluster_embeddings(self,
                            umap_embeddings: np.ndarray,
                            documents: pd.DataFrame,
                            partial_fit: bool = False,
                            y: np.ndarray = None) -> Tuple[pd.DataFrame,
                                                           np.ndarray]:
        """ Cluster UMAP embeddings with HDBSCAN

        Arguments:
            umap_embeddings: The reduced sentence embeddings with UMAP
            documents: Dataframe with documents and their corresponding IDs
            partial_fit: Whether to run `partial_fit` for online learning

        Returns:
            documents: Updated dataframe with documents and their corresponding IDs
                       and newly added Topics
            probabilities: The distribution of probabilities
        """
        if partial_fit:
            self.hdbscan_model = self.hdbscan_model.partial_fit(umap_embeddings)
            labels = self.hdbscan_model.labels_
            documents['Topic'] = labels
            self.topics_ = labels
        else:
            try:
                self.hdbscan_model.fit(umap_embeddings, y=y)
            except TypeError:
                self.hdbscan_model.fit(umap_embeddings)

            try:
                labels = self.hdbscan_model.labels_
            except AttributeError:
                labels = y
            documents['Topic'] = labels
            self._update_topic_size(documents)

        # Some algorithms have outlier labels (-1) that can be tricky to work
        # with if you are slicing data based on that labels. Therefore, we
        # track if there are outlier labels and act accordingly when slicing.
        self._outliers = 1 if -1 in set(labels) else 0

        # Extract probabilities
        probabilities = None
        if hasattr(self.hdbscan_model, "probabilities_"):
            probabilities = self.hdbscan_model.probabilities_

            if self.calculate_probabilities and is_supported_hdbscan(self.hdbscan_model):
                probabilities = hdbscan_delegator(self.hdbscan_model, "all_points_membership_vectors")

        if not partial_fit:
            self.topic_mapper_ = TopicMapper(self.topics_)
        logger.info("Clustered reduced embeddings")
        return documents, probabilities
    def _extract_words_per_topic(self,
                                 words: List[str],
                                 documents: pd.DataFrame,
                                 c_tf_idf: csr_matrix = None) -> Mapping[str,
                                                                         List[Tuple[str, float]]]:
        """ Based on tf_idf scores per topic, extract the top n words per topic

        If the top words per topic need to be extracted, then only the `words` parameter
        needs to be passed. If the top words per topic in a specific timestamp, then it
        is important to pass the timestamp-based c-TF-IDF matrix and its corresponding
        labels.

        Arguments:
            words: List of all words (sorted according to tf_idf matrix position)
            documents: DataFrame with documents and their topic IDs
            c_tf_idf: A c-TF-IDF matrix from which to calculate the top words

        Returns:
            topics: The top words per topic
        """
        if c_tf_idf is None:
            c_tf_idf = self.c_tf_idf_

        labels = sorted(list(documents.Topic.unique()))
        labels = [int(label) for label in labels]

        # Get at least the top 30 indices and values per row in a sparse c-TF-IDF matrix
        ##This is the number of words representing each topic
        top_n_words = max(self.top_n_words, 30)
        indices = []
        ####add manually
        for le, ri in zip(c_tf_idf.indptr[:-1], c_tf_idf.indptr[1:]):
            n_row_pick = min(top_n_words, ri - le)
            values = c_tf_idf.indices[le + np.argpartition(c_tf_idf.data[le:ri], -n_row_pick)[-n_row_pick:]]
            values = [values[index] if len(values) >= index + 1 else None for index in range(top_n_words )]
            indices.append(values)
        indices = np.array(indices)
        #indices = self._top_n_idx_sparse(matrix=c_tf_idf, n=top_n_words)
        #scores = self._top_n_values_sparse(matrix=c_tf_idf, indices=indices)
        top_values = []
        for row, values in enumerate(indices):
            scores = np.array([c_tf_idf[row, value] if value is not None else 0 for value in values])
            top_values.append(scores)
        scores = np.array(top_values)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top 30 words per topic based on c-TF-IDF score
        topics = {label: [(words[word_index], score)
                          if word_index is not None and score > 0
                          else ("", 0.00001)
                          for word_index, score in zip(indices[index][::-1], scores[index][::-1])
                          ]
                  for index, label in enumerate(labels)}

        # Fine-tune the topic representations
        if isinstance(self.representation_model, list):
            for tuner in self.representation_model:
                topics = tuner.extract_topics(self, documents, c_tf_idf, topics)
        elif isinstance(self.representation_model, BaseRepresentation):
            topics = self.representation_model.extract_topics(self, documents, c_tf_idf, topics)

        topics = {label: values[:self.top_n_words] for label, values in topics.items()}

        return topics
    def _top_n_values_sparse(matrix: csr_matrix, indices: np.ndarray) -> np.ndarray:
        """ Return the top n values for each row in a sparse matrix

        Arguments:
            matrix: The sparse matrix from which to get the top n indices per row
            indices: The top n indices per row

        Returns:
            top_values: The top n scores per row
        """
        top_values = []
        for row, values in enumerate(indices):
            scores = np.array([matrix[row, value] if value is not None else 0 for value in values])
            top_values.append(scores)
        return np.array(top_values)
    def _top_n_idx_sparse(matrix: csr_matrix, n: int) -> np.ndarray:
        """ Return indices of top n values in each row of a sparse matrix

        Retrieved from:
            https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix

        Arguments:
            matrix: The sparse matrix from which to get the top n indices per row
            n: The number of highest values to extract from each row

        Returns:
            indices: The top n indices per row
        """
        indices = []
        for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            n_row_pick = min(n, ri - le)
            values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
            values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
            indices.append(values)
        return np.array(indices)
    def get_topic(self, topic: int) -> Union[Mapping[str, Tuple[str, float]], bool]:
        """ Return top n words for a specific topic and their c-TF-IDF scores

        Arguments:
            topic: A specific topic for which you want its representation

        Returns:
            The top n words for a specific word and its respective c-TF-IDF scores

        Examples:

        ```python
        topic = topic_model.get_topic(12)
        ```
        """
        check_is_fitted(self)
        if topic in self.topic_representations_:
            return self.topic_representations_[topic]
        else:
            return False
    def _create_topic_vectors(self):
        """ Creates embeddings per topics based on their topic representation

        We start by creating embeddings out of the topic representation. This
        results in a number of embeddings per topic. Then, we take the weighted
        average of embeddings in a topic by their c-TF-IDF score. This will put
        more emphasis to words that represent a topic best.

        Only allow topic vectors to be created if there are no custom embeddings and therefore
        a sentence-transformer model to be used or there are custom embeddings but it is allowed
        to use a different multi-lingual sentence-transformer model
        """
        if self.embedding_model is not None and type(self.embedding_model) is not BaseEmbedder:
            topic_list = list(self.topic_representations_.keys())
            topic_list.sort()

            # Only extract top n words
            n = len(self.topic_representations_[topic_list[0]])
            if self.top_n_words < n:
                n = self.top_n_words

            # Extract embeddings for all words in all topics
            topic_words = [self.get_topic(topic) for topic in topic_list]
            topic_words = [word[0] for topic in topic_words for word in topic]
            embeddings = self._extract_embeddings(topic_words,
                                                  method="word",
                                                  verbose=False)

            # Take the weighted average of word embeddings in a topic based on their c-TF-IDF value
            # The embeddings var is a single numpy matrix and therefore slicing is necessary to
            # access the words per topic
            topic_embeddings = []
            for i, topic in enumerate(topic_list):
                word_importance = [val[1] for val in self.get_topic(topic)]
                if sum(word_importance) == 0:
                    word_importance = [1 for _ in range(len(self.get_topic(topic)))]
                topic_embedding = np.average(embeddings[i * n: n + (i * n)], weights=word_importance, axis=0)
                topic_embeddings.append(topic_embedding)

            self.topic_embeddings_ = topic_embeddings
class TopicMapper:
    """ Keep track of Topic Mappings

    The number of topics can be reduced
    by merging them together. This mapping
    needs to be tracked in BERTopic as new
    predictions need to be mapped to the new
    topics.

    These mappings are tracked in the `self.mappings_`
    attribute where each set of topic are stacked horizontally.
    For example, the most recent topics can be found in the
    last column. To get a mapping, simply take the two columns
    of topics.

    In other words, it is represented as graph:
    Topic 1 --> Topic 11 --> Topic 4 --> etc.

    Attributes:
        self.mappings_ (np.ndarray) : A  matrix indicating the mappings from one topic
                                      to another. The columns represent a collection of topics
                                      at any time. The last column represents the current state
                                      of topics and the first column represents the initial state
                                      of topics.
    """
    def __init__(self, topics: List[int]):
        """ Initalization of Topic Mapper

        Arguments:
            topics: A list of topics per document
        """
        base_topics = np.array(sorted(set(topics)))
        topics = base_topics.copy().reshape(-1, 1)
        self.mappings_ = np.hstack([topics.copy(), topics.copy()]).tolist()

    def get_mappings(self, original_topics: bool = True) -> Mapping[int, int]:
        """ Get mappings from either the original topics or
        the second-most recent topics to the current topics

        Arguments:
            original_topics: Whether we want to map from the
                             original topics to the most recent topics
                             or from the second-most recent topics.

        Returns:
            mappings: The mappings from old topics to new topics

        Examples:

        To get mappings, simply call:
        ```python
        mapper = TopicMapper(hdbscan_model)
        mappings = mapper.get_mappings(original_topics=False)
        ```
        """
        if original_topics:
            mappings = np.array(self.mappings_)[:, [0, -1]]
            mappings = dict(zip(mappings[:, 0], mappings[:, 1]))
        else:
            mappings = np.array(self.mappings_)[:, [-3, -1]]
            mappings = dict(zip(mappings[:, 0], mappings[:, 1]))
        return mappings

    def add_mappings(self, mappings: Mapping[int, int]):
        """ Add new column(s) of topic mappings

        Arguments:
            mappings: The mappings to add
        """
        for topics in self.mappings_:
            topic = topics[-1]
            if topic in mappings:
                topics.append(mappings[topic])
            else:
                topics.append(-1)

    def add_new_topics(self, mappings: Mapping[int, int]):
        """ Add new row(s) of topic mappings

        Arguments:
            mappings: The mappings to add
        """
        length = len(self.mappings_[0])
        for key, value in mappings.items():
            to_append = [key] + ([None] * (length-2)) + [value]
            self.mappings_.append(to_append)
