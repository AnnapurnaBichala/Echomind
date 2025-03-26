import torch
import torch.nn.functional as F
import json

import utils
from models import LightGCN 
import world

# Load dataset
weights = "/Users/shayan/Desktop/Echomind/code/src/deep-learning/code/checkpoints/lgn_amazon-electronics_layers-4_latent_dim-128_bpr_batch_size-2048_dropout-0_keep_prob-0.6_A_n_fold-100_test_u_batch_size-100_lr-0.001_decay-1e-06_seed-2020.pt"
dataset = utils.get_dataset(world.DATA_PATH, "amazon-electronics")

config = world.config
config["lightGCN_n_layers"] = 5
config["latent_dim_rec"] = 196

# Initialize model
model = LightGCN(config, dataset)
model = model.to(world.device)

# Load trained weights
checkpoint = torch.load(weights, map_location=torch.device(world.device))
model.load_state_dict(checkpoint["state_dict"])

# Load item ID mappings
item_mapping_path = "/Users/shayan/Desktop/test/Echomind/code/src/deep-learning/data/amazon-electronics/item_num2id.json"
with open(item_mapping_path) as f:
    item_num2id = json.load(f)

# Invert mapping for lookup
item_id2num = {asin: idx for idx, asin in enumerate(item_num2id)}


def recommend_from_items(item1_id, item2_id, top_k = 5):
    model.eval()
    with torch.no_grad():
        _, item_embeddings = model()

        emb1 = item_embeddings[item1_id]
        emb2 = item_embeddings[item2_id]

        hybrid_vec = (emb1 + emb2) / 2
        hybrid_vec = F.normalize(hybrid_vec.unsqueeze(0), dim = 1)
        item_embeddings_norm = F.normalize(item_embeddings, dim = 1)

        scores = torch.matmul(hybrid_vec, item_embeddings_norm.T).squeeze(0)

        top_indices = torch.topk(scores, k = top_k + 2).indices
        recommended = [i.item() for i in top_indices if i.item() not in [item1_id, item2_id]][:top_k]

    print("Top items:")
    for item_id in recommended:
        print(item_num2id[item_id])  # index -> asin

    return recommended
