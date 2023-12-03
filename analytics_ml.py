from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Function to recommend similar videos
def recommend_similar_videos(video_id, data, top_n=5):
    # Check if the video ID exists in the dataset
    if video_id not in data['id'].values:
        return "Video ID not found in the dataset."

    # Extract the target video's data
    target_video_data = data[data['id'] == video_id]

    # Combine title, description, and tags into a single content column
    data['content'] = data['title'] + ' ' + data['description'].fillna('') + ' ' + data['tags'].fillna('')

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['content'])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get index of the video that matches the video ID
    idx = data.index[data['id'] == video_id].tolist()[0]

    # Get the pairwise similarity scores of all videos with that video
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the videos based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top_n most similar videos
    sim_scores = sim_scores[1:top_n+1]

    # Get the video indices
    video_indices = [i[0] for i in sim_scores]

    # Return the top n most similar videos
    return data.iloc[video_indices]

if __name__ == '__main__':
    video_data = pd.read_csv('video_detail.csv')
    # Recommending similar videos for the given video ID
    recommended_videos = recommend_similar_videos("myNjmnvI6x0", video_data)
    recommended_videos[['id', 'title', 'author', 'url', 'category']] if isinstance(recommended_videos, pd.DataFrame) else recommended_videos
    print(recommended_videos)

