
Topic Modeling on Trump’s Tweet

This project requires openai==0.27.3, octis==1.10.2 and bertopic==0.14.1

##If you hit dependence conflicts regarding ocis and other packages, instsall the latest ocis with -e flag to workaround the issue

Replace variable "openai.api_key"'s value with your openai api key in create_embedding.py

Running create_embedding.py will output a trump.txt, a trump folding containing corups.tsv, and trump_twitter_np_ada.npy

Note, this may take a long time if you are on free account

After that, you may want to change the variable "filename"'s value to "trump_twitter_np_ada.npy" in the main.py if you choose to save the embedding file to other path. After executing the script, it will show results on different runs with their corresponding TC and TD.

This script will also save all the results in json format under "result" subdirectory

To calculate the average performance of runs, use sum_field.py after changing the variable "file_pattern" if neccessary.

To visualize the vectors after UMAP and Clustering, use Visualization.ipynb under visual_notebook folder. It can be run on 
