import json
import glob

##Script used to calculate end average end results

average_npmi = 0
average_diversity = 0
count = 0
file_pattern = 'result/BERTopic_sentence-t5-large_trump_*.json'
file_list = glob.glob(file_pattern)

for filename in file_list:
    path =filename
    print(path)
    with open(filename, "r") as f:
        data = json.load(f)
    for d in data:
        average_npmi += d["Scores"]["npmi"]
        average_diversity += d["Scores"]["diversity"]
        count += 1

if count > 0:
    average_npmi /= count
    average_diversity /= count

print(f"Average npmi score: {average_npmi}")
print(f"Average diversity score: {average_diversity}")