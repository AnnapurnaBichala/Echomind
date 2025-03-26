import json

# Load the JSON data from file
with open("/Users/shayan/Desktop/Echomind/code/src/deep-learning/data/amazon-electronics/validation_data.json", "r") as json_file:
    data = json.load(json_file)

# Write the formatted text output
with open("/Users/shayan/Desktop/Echomind/code/src/deep-learning/data/amazon-electronics/validation.txt", "w") as txt_file:
    for idx, sublist in enumerate(data):
        line = f"{idx} " + " ".join(map(str, sublist)) + "\n"
        txt_file.write(line)