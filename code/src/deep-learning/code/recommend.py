import torch
import torch.nn.functional as F

weights = "/Users/shayan/Downloads/LightGCN-main/code/checkpoints/lgn_amazon-electro_layers-4_latent_dim-128_bpr_batch_size-2048_dropout-0_keep_prob-0.6_A_n_fold-100_test_u_batch_size-100_lr-0.01_decay-1e-05_seed-2020.pt"
dataset = utils.get_dataset(world.DATA_PATH, "amazon-electro")

config = world.config
config["lightGCN_n_layers"] = 4
config["latent_dim_rec"] = 128

model = LightGCN(world.config, dataset)
model = model.to(world.device)

checkpoint = torch.load(weights, map_location = torch.device(world.device))

model.load_state_dict(checkpoint["state_dict"])

def recommend_from_items(item1, item2, top_k=5):
    model.eval()
    with torch.no_grad():
        _, item_embeddings = model()

        emb1 = item_embeddings[item1]
        emb2 = item_embeddings[item2]

        hybrid_vec = (emb1 + emb2) / 2
        hybrid_vec = F.normalize(hybrid_vec.unsqueeze(0), dim = 1)
        item_embeddings_norm = F.normalize(item_embeddings, dim = 1)

        scores = torch.matmul(hybrid_vec, item_embeddings_norm.T).squeeze(0)

        top_indices = torch.topk(scores, k = top_k + 2).indices
        recommended = [i.item() for i in top_indices if i.item() not in [item1, item2]][:top_k]

    return recommended