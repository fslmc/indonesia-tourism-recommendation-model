import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# 3. Hybrid Model
class HybridRecommender:
    def __init__(self, content_weight=0.6, collab_weight=0.4):
        self.content_model = NearestNeighbors(metric='cosine')
        self.collab_model = NearestNeighbors(metric='cosine')
        self.weights = [content_weight, collab_weight]
        self.content_features = None
        self.collab_matrix = None
        
    def fit(self, content_features, collab_matrix):
        # Simpan data training
        self.content_features = content_features
        self.collab_matrix = collab_matrix
        
        # Train model
        self.content_model.fit(content_features)
        self.collab_model.fit(collab_matrix)
        
    def recommend(self, user_idx, k=10):
        # Content-based
        content_dist, content_idx = self.content_model.kneighbors(
            [self.content_features[user_idx]], 
            n_neighbors=k*2
        )
        
        # Collaborative
        collab_dist, collab_idx = self.collab_model.kneighbors(
            [self.collab_matrix.iloc[user_idx].values],
            n_neighbors=k*2
        )
        
        # Hybrid scoring
        combined = pd.DataFrame({
            'place_id': np.concatenate([content_idx[0], collab_idx[0]]),
            'score': np.concatenate([
                1 - content_dist[0] * self.weights[0],
                1 - collab_dist[0] * self.weights[1]
            ])
        }).sort_values('score', ascending=False).head(k)
        
        return combined['place_id'].tolist()

# 2. Collaborative Filtering Components
def prepare_collab_data():
    # Create user-item matrix
    click_counts = clicks.groupby(['User_Id', 'Place_Id']).size().unstack(fill_value=0)
    search_counts = searches.groupby(['User_Id', 'Implied_Place_Id']).size().unstack(fill_value=0)
    
    # Combine interactions
    interaction_matrix = click_counts.add(search_counts, fill_value=0)
    return interaction_matrix.fillna(0)

# 1. Content-Based Filtering Components
def prepare_content_features():
    # User features
    mlb = MultiLabelBinarizer()
    user_cats = mlb.fit_transform(users['Preferred_Categories'].apply(eval))
    user_tags = mlb.fit_transform(users['Interest_Tags'].apply(eval))
    
    # Place features
    place_cats = mlb.fit_transform(places['Category'].apply(lambda x: [x]))
    place_desc_tfidf = TfidfVectorizer().fit_transform(places['Description'])
    
    return {
        'user_features': np.hstack([user_cats, user_tags]),
        'place_features': np.hstack([place_cats, place_desc_tfidf.toarray()]),
        'transformers': mlb
    }