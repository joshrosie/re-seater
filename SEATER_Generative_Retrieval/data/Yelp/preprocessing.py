# %%
import pandas as pd
import numpy as np
import os
import polars as pl

# Load Yelp data from raw_data directory
raw_data_path = '../../../../Yelp/Yelp/raw_data/'
filter_size = 5

# %%
review_df = pd.read_json(os.path.join(raw_data_path, 'yelp_academic_dataset_review.json'), lines=True)

# %%
# Create business_id to incremental id mapping
unique_business_ids = review_df['business_id'].unique()
business_id_to_idx = {business_id: idx for idx, business_id in enumerate(unique_business_ids)}

# Create user_id to incremental id mapping
unique_user_ids = review_df['user_id'].unique()
user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}


# %%
# Filter items by popularity
item_counts = review_df['business_id'].value_counts()
popular_items = item_counts[item_counts >= filter_size].index.tolist()

# Filter reviews to only include popular items
filtered_reviews = review_df[review_df['business_id'].isin(popular_items)]

# Filter users by activity
user_counts = filtered_reviews['user_id'].value_counts()
active_users = user_counts[user_counts >= filter_size].index.tolist()

# Filter reviews to only include active users
filtered_reviews = filtered_reviews[filtered_reviews['user_id'].isin(active_users)]

# Get unique user IDs and shuffle them
user_ids = filtered_reviews['user_id'].unique()
np.random.shuffle(user_ids)

# Split users into train/valid/test
num_users = len(user_ids)
split_1 = int(num_users * 0.8)
split_2 = int(num_users * 0.9)
train_users = user_ids[:split_1]
valid_users = user_ids[split_1:split_2]
test_users = user_ids[split_2:]

# Process each split
train_data = filtered_reviews[filtered_reviews['user_id'].isin(train_users)]
valid_data = filtered_reviews[filtered_reviews['user_id'].isin(valid_users)]
test_data = filtered_reviews[filtered_reviews['user_id'].isin(test_users)]

# %%
# display(valid_data.head())

# %%
# Process validation data
valid_formatted = pd.DataFrame(columns=['user_id', 'given_user_history', 'predicting_items'])

for user_id, group in valid_data.groupby('user_id'):
    # Sort by date first
    sorted_group = group.sort_values('date')
    business_ids = sorted_group['business_id'].tolist()
    split_idx = int(len(business_ids) * 0.8)
    given_history = [business_id_to_idx[bid] for bid in business_ids[:split_idx]]
    predicting_items = [business_id_to_idx[bid] for bid in business_ids[split_idx:]]
    
    new_row = pd.DataFrame({
        'user_id': [user_id_to_idx[user_id]],
        'given_user_history': [given_history],
        'predicting_items': [predicting_items]
    })
    valid_formatted = pd.concat([valid_formatted, new_row], ignore_index=True)

# Process test data
test_formatted = pd.DataFrame(columns=['user_id', 'given_user_history', 'predicting_items'])

for user_id, group in test_data.groupby('user_id'):
    # Sort by date first
    sorted_group = group.sort_values('date')
    business_ids = sorted_group['business_id'].tolist()
    split_idx = int(len(business_ids) * 0.8)
    given_history = [business_id_to_idx[bid] for bid in business_ids[:split_idx]]
    predicting_items = [business_id_to_idx[bid] for bid in business_ids[split_idx:]]
    
    new_row = pd.DataFrame({
        'user_id': [user_id_to_idx[user_id]],
        'given_user_history': [given_history],
        'predicting_items': [predicting_items]
    })
    test_formatted = pd.concat([test_formatted, new_row], ignore_index=True)

# Process training data
train_formatted = pd.DataFrame(columns=['uid', 'his_seq', 'next_item'])

for user_id, group in train_data.groupby('user_id'):
    # Sort by date first
    sorted_group = group.sort_values('date')
    business_ids = sorted_group['business_id'].tolist()
    
    # Create all possible sequences
    for i in range(1, len(business_ids)):
        history = [business_id_to_idx[bid] for bid in business_ids[:i]]
        target = business_id_to_idx[business_ids[i]]
        
        new_row = pd.DataFrame({
            'uid': [user_id_to_idx[user_id]],
            'his_seq': [history],
            'next_item': [target]
        })
        train_formatted = pd.concat([train_formatted, new_row], ignore_index=True)


# %%
# write to file
train_formatted.to_csv('dataset/training.tsv', sep='\t', index=False)
valid_formatted.to_csv('dataset/validation.tsv', sep='\t', index=False)
test_formatted.to_csv('dataset/test.tsv', sep='\t', index=False)

# display(train_formatted.head())


# %%



