import shelve
import sys

# # Directory where your .db files are stored
db_path = "z_ckpts/tqhxk9vc/train_tid_2_items.db"  # Replace with your actual directory path
# total_size = 0  # Variable to accumulate total memory usage
# with shelve.open(db_path) as db:
#     count = 0
#     for fid, group_scores in db.items():
#         # print(fid)
#         for i in group_scores:
#             print(i)
#         # print(group_scores)
#         # print(len(group_scores))
#         break

# # Directory where your .db files are stored
db_path = "z_ckpts/tqhxk9vc/train_fid_xmodal_scores.db"  # Replace with your actual directory path
total_size = 0  # Variable to accumulate total memory usage
with shelve.open(db_path) as db:
    count = 0
    for fid, group_scores in db.items():
        # print(fid)
        for i in group_scores:
            print(i)
        # print(group_scores)
        # print(len(group_scores))
        break

# fid_score_fpath = os.path.join(ckp_fpath, f"{name}_fid_xmodal_scores.db")
#     tid_score_fpath = os.path.join(ckp_fpath, f"{name}_tid_xmodal_scores.db")

#     with shelve.open(filename=fid_score_fpath, flag="n", protocol=2) as fid_stream, \
#         shelve.open(filename=tid_score_fpath, flag="n", protocol=2) as tid_stream:

#         # Iterate through unique audio file identifiers
#         for fid in tqdm(ds.text_data["fid"].unique(), desc=f"Computing cross-modal scores for {name} dataset"):
#             # Initialize dictionaries to store scores
#             fid_group_scores = {}  # Indexed by fid
#             tid_group_scores = {}  # Indexed by tid

#             # Encode audio data
#             audio_vec = torch.as_tensor(ds.audio_data[fid][()]).to(device)
#             audio_vec = torch.unsqueeze(audio_vec, dim=0)
#             audio_embed = model.audio_branch(audio_vec)[0]  # 300 is its shape

#             # For a single fid, calculate the scores for all tids
#             for i in range(0, len(text2vec), batch_size):
#                 # Get batch of text IDs
#                 batch_text_ids = list(text2vec.keys())[i:min(i + batch_size, len(text2vec))]

#                 # Create a tensor of embedded text vectors for the batch
#                 batch_text_embeds = torch.stack([text2vec[tid] for tid in batch_text_ids]).to(device)

#                 # Reshape for matrix multiplication (batch_size, embedding_dim)
#                 batch_text_embeds = batch_text_embeds.reshape(-1, batch_text_embeds.shape[-1])

#                 # Calculate cross-modal scores for the batch
#                 xmodal_scores = criterion_utils.score(audio_embed, batch_text_embeds, obj_params["args"].get("dist", "dot_product"))

#                 # Update both dictionaries
#                 for j, tid in enumerate(batch_text_ids):
#                     fid_group_scores[tid] = xmodal_scores[j].item()
#                     if tid not in tid_group_scores:
#                         tid_group_scores[tid] = {}
#                     tid_group_scores[tid][fid] = xmodal_scores[j].item()

#             # Save scores for the current audio file (fid)
#             fid_stream[fid] = fid_group_scores

#         # Save the tid-based scores after processing all fids
#         tid_stream.update(tid_group_scores)

#     print("Saved:", fid_score_fpath, "and", tid_score_fpath)
