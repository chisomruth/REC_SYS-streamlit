import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv
import pydeck as pdk

# Load API Key from .env file
load_dotenv('credentials.env')
api_key = os.getenv("GOOGLE_API_KEY")

# Load the dataset
df = pd.read_csv(r"C:\Users\chiso\Downloads\cleaned_health_data.csv")

# Ensure latitude and longitude are numeric
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Remove rows with missing or invalid coordinates
df = df.dropna(subset=['latitude', 'longitude'])

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.05, min_samples=5)
df['Cluster'] = dbscan.fit_predict(df[['latitude', 'longitude']])


def geocode_address_google(user_address):
    """Geocodes the user's address using Google Maps Geocoding API."""
    user_address = quote_plus(user_address)  # URL-encode the address
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={user_address}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            location = results[0]['geometry']['location']
            return location['lat'], location['lng']
    return None, None


def find_nearest_cluster(user_lat, user_lon):
    """Find the nearest cluster to the user's location."""
    if df['Cluster'].nunique() == 0:
        return None  # No clusters available
    cluster_centers = df.groupby('Cluster')[['latitude', 'longitude']].mean()
    distances = np.sqrt((cluster_centers['latitude'] - user_lat) ** 2 +
                        (cluster_centers['longitude'] - user_lon) ** 2)
    return distances.idxmin()


def recommend_hospitals_cluster_knn(user_lat, user_lon, n_recommendations=5, min_rating=0):
    """Recommend hospitals using a hybrid of clustering and KNN with rating filters."""
    nearest_cluster = find_nearest_cluster(user_lat, user_lon)

    if nearest_cluster is None or nearest_cluster not in df['Cluster'].unique():
        return pd.DataFrame()

    cluster_hospitals = df[df['Cluster'] == nearest_cluster]

    # Apply minimum rating filter
    cluster_hospitals = cluster_hospitals[cluster_hospitals['ratings'] >= min_rating]

    if cluster_hospitals.empty:
        return pd.DataFrame()

    # KNN within the cluster
    knn = NearestNeighbors(n_neighbors=min(len(cluster_hospitals), n_recommendations), algorithm='auto')
    knn.fit(cluster_hospitals[['latitude', 'longitude']])

    user_location = pd.DataFrame([[user_lat, user_lon]], columns=['latitude', 'longitude'])
    distances, indices = knn.kneighbors(user_location)

    nearest_hospitals = cluster_hospitals.iloc[indices[0]].copy()

    # Sort by ratings
    nearest_hospitals = nearest_hospitals.sort_values(by='ratings', ascending=False)

    # Add Google Maps link
    nearest_hospitals['google_maps_link'] = nearest_hospitals.apply(
        lambda row: f"https://www.google.com/maps?q={row['latitude']},{row['longitude']}", axis=1
    )

    return nearest_hospitals


# Streamlit App
st.set_page_config(page_title="Hospital Recommender", page_icon="🏥", layout="wide")

st.title("🏥 Hospital Location-Based Recommender System")
st.markdown("""
This interactive app helps you find the **best hospitals** near your location.  
Just enter your address, apply filters, and we'll recommend the nearest options with ratings and Google Maps links.  
""")

# User input
user_address = st.text_input("📍 Enter your full address:")
n_recommendations = st.slider("Select the number of recommendations:", 1, 10, 5)
min_rating = st.slider("Minimum hospital rating (0-5):", 0.0, 5.0, 0.0, step=0.5)

if st.button("🔍 Find Hospitals"):
    if user_address:
        with st.spinner("Fetching your location and finding the best hospitals..."):
            # Geocode the user address
            user_lat, user_lon = geocode_address_google(user_address)

            if user_lat is None or user_lon is None:
                st.error("❌ Invalid address or geocoding failed. Please try again.")
            else:
                # Get recommendations
                recommendations = recommend_hospitals_cluster_knn(
                    user_lat, user_lon, n_recommendations=n_recommendations, min_rating=min_rating
                )

                if recommendations.empty:
                    st.warning("⚠️ No hospitals found in the vicinity with the specified filters.")
                else:
                    st.subheader("🏥 Recommended Hospitals")
                    for _, hospital in recommendations.iterrows():
                        with st.expander(f"✨ {hospital['facility_name']}"):
                            st.markdown(f"""
                            **Address:** {hospital['address']}  
                            **Ratings:** {hospital['ratings']} ⭐  
                            [🌐 View on Google Maps]({hospital['google_maps_link']})  
                            """)

                    # Map Visualization
                    st.subheader("🗺️ Map of Recommended Hospitals")
                    hospital_map = pdk.Deck(
                        initial_view_state=pdk.ViewState(
                            latitude=user_lat,
                            longitude=user_lon,
                            zoom=12,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=recommendations,
                                get_position='[longitude, latitude]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=200,
                            ),
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=pd.DataFrame([[user_lon, user_lat]], columns=["longitude", "latitude"]),
                                get_position='[longitude, latitude]',
                                get_color='[0, 150, 0, 200]',
                                get_radius=300,
                            ),
                        ],
                        tooltip={"text": "{facility_name}\n{address}\nRatings: {ratings} ⭐"},
                    )
                    st.pydeck_chart(hospital_map)
    else:
        st.error("❌ Please enter your address.")

# Add a footer
st.markdown("---")
st.markdown("""
### 📌 Note:
* Ensure your address is entered correctly.
* Use the filters to refine your search by rating and number of recommendations.
""")
