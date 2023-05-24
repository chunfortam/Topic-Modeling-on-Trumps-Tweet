# CS_274_Topic_Modeling
 Topic modeling for social media

Running the code:
Download the repo from https://github.com/MaartenGr/BERTopic_evaluation
pip install the destination folder
##If you hit dependence conflicts regarding ocis, instsall ocis while with -e flag to workaround

Replace variable "openai.api_key"'s value with your openai api key in create_embedding.py
Run the scripts and it will output a trump.txt, a trump folding containing corups.tsv, and trump_twitter_np_ada.npy
Note, this may take 24hr+ if you are on free account

After that, you want to change the variable "filename"'s value to "trump_twitter_np_ada.npy". It should show results on different runs with their corresponding TC and TD.
This script will output all the results in json format under "result" subdirectory

To calculate the average performance of runs, use sum_field.py

To visualize the vectors after UMAP and Clustering, use Visualization.ipynb under visual_notebook folder
