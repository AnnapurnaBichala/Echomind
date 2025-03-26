import torch
import torch.nn.functional as F
import json

import utils
from models import LightGCN 
import world


dataset = utils.get_dataset(world.DATA_PATH, "amazon-electronics")
config = world.config
config["lightGCN_n_layers"] = 4
config["latent_dim_rec"] = 128


item_mapping_path = "../data/amazon-electronics/item_num2id.json"
with open(item_mapping_path) as f:
    item_num2id = json.load(f)
item_id2num = {asin: idx for idx, asin in enumerate(item_num2id)}


def load_model(weights_path):
    model = LightGCN(config, dataset).to(world.device)
    checkpoint = torch.load(weights_path, map_location=torch.device(world.device))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def recommend_from_items(model, item1_id, item2_id, top_k = 3):
    with torch.no_grad():
        _, item_embeddings = model()

        emb1 = item_embeddings[item1_id]
        emb2 = item_embeddings[item2_id]

        hybrid_vec = (emb1 + emb2) / 2
        hybrid_vec = F.normalize(hybrid_vec.unsqueeze(0), dim=1)
        item_embeddings_norm = F.normalize(item_embeddings, dim=1)

        scores = torch.matmul(hybrid_vec, item_embeddings_norm.T).squeeze(0)

        top_indices = torch.topk(scores, k=top_k + 2).indices
        recommended = [item_num2id[i.item()] for i in top_indices if i.item() not in [item1_id, item2_id]][:top_k]

    return recommended