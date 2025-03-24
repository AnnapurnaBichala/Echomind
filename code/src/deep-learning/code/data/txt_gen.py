import json

# Load the JSON data from file
with open("/Users/shayan/Downloads/LightGCN-main/data/amazon-electronics/test_data.json", "r") as json_file:
    data = json.load(json_file)

# Write the formatted text output
with open("/Users/shayan/Downloads/LightGCN-main/data/amazon-electronics/test.txt", "w") as txt_file:
    for idx, sublist in enumerate(data):
        line = f"{idx} " + " ".join(map(str, sublist)) + "\n"
        txt_file.write(line)