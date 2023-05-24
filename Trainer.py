##The following class is a modified version of Trainer class from  BERTopic_evaluation
#https://github.com/MaartenGr/BERTopic_evaluation
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
import itertools
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable
import json
from GPTopic import GPTopic
import numpy as np
import time
class Trainer:
    """Train and evaluate a topic model

    Arguments:
        dataset: The dataset to be used, should be a string and either a
                 dataset found in OCTIS or a custom dataset
        model_name: The name of the topic model to be used:
                        * BERTopic
                        * Top2Vec
                        * CTM_CUSTOM (original package)
                        * ETM (OCTIS)
                        * LDA (OCTIS)
                        * CTM (OCTIS)
                        * NMF (OCTIS)
        params: The parameters of the model to be used
        topk: The top n words in each topic to include
        custom_dataset: Whether a custom dataset is used
        bt_embeddings: Pre-trained embeddings used in BERTopic to speed
                       up training.
        bt_timestamps: Timestamps used in BERTopic for dynamic
                       topic modeling
        bt_nr_bins: Number of bins to create from timestamps in BERTopic
        custom_model: A custom BERTopic or Top2Vec class
        verbose: Control the verbosity of the trainer

    Usage:

    ```python
    from evaluation import Trainer
    dataset, custom = "20NewsGroup", False
    params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": 42}

    trainer = Trainer(dataset=dataset,
                      model_name="LDA",
                      params=params,
                      custom_dataset=custom,
                      verbose=True)
    ```

    Note that we need to specify whether a custom OCTIS dataset is used.
    Since we use a preprocessed dataset from OCTIS [here](https://github.com/MIND-Lab/OCTIS#available-datasets),
    no custom dataset is used.

    This trainer focused on iterating over all combinations of parameters in `params`.
    In the example above, we iterate over different number of topics.
    """

    def __init__(
        self,
        dataset: str,
        model_name: str,
        params: Mapping[str, Any],
        topk: int = 10,
        custom_dataset: bool = False,
        bt_embeddings: np.ndarray = None,
        bt_timestamps: List[str] = None,
        bt_nr_bins: int = None,
        custom_model=None,
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.custom_dataset = custom_dataset
        self.model_name = model_name
        self.params = params
        self.topk = topk
        self.timestamps = bt_timestamps
        self.nr_bins = bt_nr_bins
        self.embeddings = bt_embeddings
        self.ctm_preprocessed_docs = None
        self.custom_model = custom_model
        self.verbose = verbose

        # Prepare data and metrics
        self.data = self.get_dataset()
        self.metrics = self.get_metrics()

        # CTM
        self.qt_ctm = None
        self.training_dataset_ctm = None

    def train(self, save: str = False) -> Mapping[str, Any]:
        """Train a topic model

        Arguments:
            save: The name of the file to save it to.
                  It will be saved as a .json in the current
                  working directory

        Usage:

        ```python
        from evaluation import Trainer
        dataset, custom = "20NewsGroup", False
        params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": 42}

        trainer = Trainer(dataset=dataset,
                        model_name="LDA",
                        params=params,
                        custom_dataset=custom,
                        verbose=True)
        results = trainer.train(save="LDA_results")
        ```
        """

        results = []

        # Loop over all parameters
        params_name = list(self.params.keys())
        params = {
            param: (value if type(value) == list else [value])
            for param, value in self.params.items()
        }
        new_params = list(itertools.product(*params.values()))
        for param_combo in new_params:

            # Train and evaluate model
            params_to_use = {
                param: value for param, value in zip(params_name, param_combo)
            }
            output, computation_time = self._train_tm_model(params_to_use)
            scores = self.evaluate(output)

            # Update results
            params_to_use.pop("hdbscan_model", None )
            result = {
                "Dataset": self.dataset,
                "Dataset Size": len(self.data.get_corpus()),
                "Model": self.model_name,
                "Params": params_to_use,
                "Scores": scores,
                "Computation Time": computation_time,
            }
            results.append(result)

        if save:
            with open(f"result/{save}.json", "w") as f:
                json.dump(results, f)

            try:
                from google.colab import files

                files.download(f"{save}.json")
            except ImportError:
                pass

        return results
    def _train_gptopic(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        data = self.data.get_corpus()
        data = [" ".join(words) for words in data]
        params["calculate_probabilities"] = False

        if self.custom_model is not None:
            model = self.custom_model(**params)
        else:
            model = GPTopic(**params)

        start = time.time()
        topics, _ = model.fit_transform(data, self.embeddings)

        # Dynamic Topic Modeling
        if self.timestamps:
            topics_over_time = model.topics_over_time(
                data,
                topics,
                self.timestamps,
                nr_bins=self.nr_bins,
                evolution_tuning=False,
                global_tuning=False,
            )
            unique_timestamps = topics_over_time.Timestamp.unique()
            dtm_topics = {}
            for unique_timestamp in unique_timestamps:
                dtm_topic = topics_over_time.loc[
                    topics_over_time.Timestamp == unique_timestamp, :
                ].sort_values("Frequency", ascending=True)
                dtm_topic = dtm_topic.loc[dtm_topic.Topic != -1, :]
                dtm_topic = [topic.split(", ") for topic in dtm_topic.Words.values]
                dtm_topics[unique_timestamp] = {"topics": dtm_topic}

                all_words = [word for words in self.data.get_corpus() for word in words]

                updated_topics = []
                for topic in dtm_topic:
                    updated_topic = []
                    for word in topic:
                        if word not in all_words:
                            print(word)
                            updated_topic.append(all_words[0])
                        else:
                            updated_topic.append(word)
                    updated_topics.append(updated_topic)

                dtm_topics[unique_timestamp] = {"topics": updated_topics}

            output_tm = dtm_topics

        end = time.time()
        computation_time = float(end - start)

        if not self.timestamps:
            all_words = [word for words in self.data.get_corpus() for word in words]
            gptopic_topics = [
                [
                    vals[0] if vals[0] in all_words else all_words[0]
                    for vals in model.get_topic(i)[:10]
                ]
                for i in range(len(set(topics)) - 1)
            ]

            output_tm = {"topics": gptopic_topics}

        return output_tm, computation_time
    def _train_tm_model(
        self, params: Mapping[str, Any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Select and train the Topic Model"""
        # Train custom CTM
        return self._train_gptopic(params)

    def evaluate(self, output_tm):
        """Using metrics and output of the topic model, evaluate the topic model"""
        if self.timestamps:
            results = {str(timestamp): {} for timestamp, _ in output_tm.items()}
            for timestamp, topics in output_tm.items():
                self.metrics = self.get_metrics()
                for scorers, _ in self.metrics:
                    for scorer, name in scorers:
                        score = scorer.score(topics)
                        results[str(timestamp)][name] = float(score)

        else:
            # Calculate results
            results = {}
            for scorers, _ in self.metrics:
                for scorer, name in scorers:
                    score = scorer.score(output_tm)
                    results[name] = float(score)

            # Print results
            if self.verbose:
                print("Results")
                print("============")
                for metric, score in results.items():
                    print(f"{metric}: {str(score)}")
                print(" ")

        return results

    def get_dataset(self):
        """Get dataset from OCTIS"""
        data = Dataset()

        if self.custom_dataset:
            data.load_custom_dataset_from_folder(self.dataset)
        else:
            data.fetch_dataset(self.dataset)
        return data

    def get_metrics(self):
        """Prepare evaluation measures using OCTIS"""
        npmi = Coherence(texts=self.data.get_corpus(), topk=self.topk, measure="c_npmi")
        topic_diversity = TopicDiversity(topk=self.topk)

        # Define methods
        coherence = [(npmi, "npmi")]
        diversity = [(topic_diversity, "diversity")]
        metrics = [(coherence, "Coherence"), (diversity, "Diversity")]

        return metrics