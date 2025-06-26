import pandas as pd
import os
import numpy as np
from collections import Counter
from datetime import timedelta
import json

print("start the preprocessing...")
dataframes = {}
parquet_directory_path = "./sequential/5b" 

for file in os.listdir(parquet_directory_path):
    if file.endswith('.parquet'):
        file_path = os.path.join(parquet_directory_path, file)
        print(file_path)
        df = pd.read_parquet(file_path)
        dataframes[file] = df

print("Parquet files loaded")

def clean_is_organic(df):
    # Remove entries where is_organic == 0 from all relevant columns
    def filter_row(row):
        mask = [flag == 1 for flag in row['is_organic']]
        row['is_organic'] = [flag for flag, keep in zip(row['is_organic'], mask) if keep]
        row['item_id'] = [item for item, keep in zip(row['item_id'], mask) if keep]
        row['timestamp'] = [ts for ts, keep in zip(row['timestamp'], mask) if keep]
        row['played_ratio_pct'] = [ratio for ratio, keep in zip(row['played_ratio_pct'], mask) if keep]

        return row

    return df.apply(filter_row, axis=1)

organic_listen = clean_is_organic(dataframes['listens.parquet'])

# Filtering for played_ratio_pct >= 50%
def filter_played_ratio(df):
    def filter_row(row):
        mask = [pr >= 50 for pr in row['played_ratio_pct']]
        row['timestamp'] = [ts for ts, keep in zip(row['timestamp'], mask) if keep]
        row['item_id'] = [item for item, keep in zip(row['item_id'], mask) if keep]
        row['played_ratio_pct'] = [pr for pr, keep in zip(row['played_ratio_pct'], mask) if keep]
        return row

    return df.apply(filter_row, axis=1)

filtered_organic_listen = filter_played_ratio(organic_listen)
filtered_organic_listen.drop(columns=['is_organic', 'track_length_seconds'])

# === CONFIG ===
DEBUG_MODE = False
DEBUG_USER_COUNT = 3

# === UTILITIES ===
def dedup(seq):
    return list(dict.fromkeys(seq)) if isinstance(seq, list) else seq

# STEP 1: Copy original data
session_df = filtered_organic_listen.copy()

# STEP 1.5: Debug mode
if DEBUG_MODE:
    sample_users = session_df['uid'].unique()[:DEBUG_USER_COUNT]
    session_df = session_df[session_df['uid'].isin(sample_users)].copy()
    print(f"🔍 DEBUG MODE: Using {len(sample_users)} users")

# STEP 2: Explode list-columns
session_df = session_df.explode(['timestamp', 'item_id', 'played_ratio_pct'])

# STEP 3: Convert timestamps
session_df['timestamp'] = pd.to_numeric(session_df['timestamp'], errors='coerce')
session_df = session_df.dropna(subset=['timestamp'])
session_df['timestamp'] = pd.to_datetime(session_df['timestamp'], unit='s')

# STEP 4: Sort
session_df = session_df.sort_values(by=['uid', 'timestamp'])

# STEP 5: Remap IDs
all_uids = session_df['uid'].dropna().unique()
all_items = session_df['item_id'].dropna().unique()

uid2idx = {str(uid): i for i, uid in enumerate(sorted(all_uids))}
item2idx = {str(item): i + 1 for i, item in enumerate(sorted(all_items))}

session_df['uid'] = session_df['uid'].map(lambda x: uid2idx.get(str(x)))
session_df['item_id'] = session_df['item_id'].map(lambda x: item2idx.get(str(x)))
session_df = session_df.dropna(subset=['uid', 'item_id'])


session_df['uid'] = session_df['uid'].astype(int)
session_df['item_id'] = session_df['item_id'].astype(int)

# STEP 6: Time ranges
min_time = session_df['timestamp'].min()
train_end = min_time + timedelta(days=299)
val_start = train_end
val_end = val_start + timedelta(days=1)
test_start = val_end + timedelta(minutes=30)
test_end = test_start + timedelta(days=1)

# STEP 7: Split data
train_data = session_df[session_df['timestamp'] < train_end]
val_data = session_df[(session_df['timestamp'] >= val_start) & (session_df['timestamp'] < val_end)]
test_data = session_df[(session_df['timestamp'] >= test_start) & (session_df['timestamp'] < test_end)]

# STEP 8: Train set (dedup first, then slice) --> Normal case
train_rows = []

for uid, group in train_data.groupby('uid'):
    items = group['item_id'].tolist()
    for i in range(1, len(items)):
        history = dedup(items[:i])[-20:]  # ✅ correct order
        next_item = items[i]
        train_rows.append({'uid': uid, 'history': history, 'next_item': next_item})

train_df = pd.DataFrame(train_rows)
# STEP 8: Train set (fixed windows of 20) ----> BERC4REC
train_rows_bert = []

for uid, group in train_data.groupby('uid'):
    items_bert = dedup(group['item_id'].tolist())
    
    # Split items into chunks of 20
    for i in range(0, len(items) - 40 + 1, 20):  # Ensure we have full 20 for history and next
        history_chunk = items[i:i+20]
        next_chunk = items[i+20:i+40]
        if len(next_chunk) == 20:  # Ensure both chunks are full
            train_rows_bert.append({
            'uid': uid,
            'history': history_chunk,  # growing history
            })

train_df_bert = pd.DataFrame(train_rows_bert)


# STEP 9: Validation set
val_rows = []

for uid, group in val_data.groupby('uid'):
    train_items = train_data[train_data['uid'] == uid]['item_id'].tolist()
    if not train_items:
        continue
    history = dedup(train_items)[-20:]
    predicting_items = dedup(group['item_id'].tolist())
    if predicting_items:
        val_rows.append({'uid': uid, 'history': history, 'predicting_items': predicting_items})

val_df = pd.DataFrame(val_rows)

# STEP 10: Test set
test_rows = []

for uid, group in test_data.groupby('uid'):
    prior_items = session_df[
        (session_df['uid'] == uid) &
        (session_df['timestamp'] < val_end)  # ✅ ends before gap
    ]['item_id'].tolist()
    if not prior_items:
        continue
    history = dedup(prior_items)[-20:]
    predicting_items = dedup(group['item_id'].tolist())
    if predicting_items:
        test_rows.append({'uid': uid, 'history': history, 'predicting_items': predicting_items})

test_df = pd.DataFrame(test_rows)

# STEP 11: Save mappings
uid2idx_str = {str(k): int(v) for k, v in uid2idx.items()}
item2idx_str = {str(k): int(v) for k, v in item2idx.items()}

with open("./dataset/uid2idx_5b.json", "w") as f:
    json.dump(uid2idx_str, f)

with open("./dataset/item2idx_5b.json", "w") as f:
    json.dump(item2idx_str, f)

# STEP 12: Save datasets
train_df.to_csv("./dataset/training_5b.tsv", sep="\t", index=False)
train_df_bert.to_csv("./dataset/training_bert4rec_5b.tsv", sep="\t", index=False)
val_df.to_csv("./dataset/validation_5b.tsv", sep="\t", index=False)
test_df.to_csv("./dataset/test_5b.tsv", sep="\t", index=False)

# STEP 13: Summary
print("✅ Done!")
print("Train size:", train_df.shape)
print("Validation size:", val_df.shape)
print("Test size:", test_df.shape)
print("Unique users:", len(uid2idx))
print("Unique items:", len(item2idx))

# STEP 14: Prediction stats
test_lengths = test_df['predicting_items'].apply(len)
print("🔍 Test predicting items:")
print("- Avg length:", test_lengths.mean())
print("- Max length:", test_lengths.max())
print("- Min length:", test_lengths.min())
