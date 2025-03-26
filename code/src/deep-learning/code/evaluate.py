import world
import procedures
import utils
from models import LightGCN
import torch

weights = "/Users/shayan/Desktop/Echomind/code/src/deep-learning/code/checkpoints/lgn_amazon-electronics_layers-4_latent_dim-128_bpr_batch_size-2048_dropout-0_keep_prob-0.6_A_n_fold-100_test_u_batch_size-100_lr-0.001_decay-1e-06_seed-2020.pt"
dataset = utils.get_dataset(world.DATA_PATH, "amazon-electronics")

config = world.config
config["lightGCN_n_layers"] = 5
config["latent_dim_rec"] = 196

model = LightGCN(world.config, dataset)
model = model.to(world.device)

checkpoint = torch.load(weights, map_location = torch.device(world.device))

model.load_state_dict(checkpoint["state_dict"])

test_metrics = procedures.eval_pairwise(dataset, model, world.config["multicore"])
num_test_users = len(dataset.test_dict)

for i_k, k in enumerate(world.topks):
    precision_k = test_metrics["precision"][i_k] / num_test_users
    recall_k = test_metrics["recall"][i_k] / num_test_users
    ndcg_k = test_metrics["ndcg"][i_k] / num_test_users

    print(f"Precision@{k}: {precision_k:.4f}")
    print(f"Recall@{k}: {recall_k:.4f}")
    print(f"NDCG@{k}: {ndcg_k:.4f}")