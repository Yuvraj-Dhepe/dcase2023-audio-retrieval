
import shelve
import sys

# Directory where your .db files are stored
db_path = "z_ckpts/vj2b6kld/train_tid_2_items_original.db"  # Replace with your actual directory path

total_size = 0  # Variable to accumulate total memory usage

with shelve.open(db_path) as db:
    count = 0
    for fid, group_scores in db.items():
        print(fid)
        for i in group_scores:
            print(i)
            break
        # print(type(group_scores))
        print(len(group_scores)) # 19195
        break

# Directory where your .db files are stored
db_path = "z_ckpts/vj2b6kld/train_tid_2_items_optimized.db"  # Replace with your actual directory path

total_size = 0  # Variable to accumulate total memory usage

with shelve.open(db_path) as db:
    count = 0
    for fid, group_scores in db.items():
        print(fid)
        for i in group_scores:
            print(i)
            break
        # print(type(group_scores))
        print(len(group_scores)) # 19195
        break