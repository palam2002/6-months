import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
from collections import Counter
import random

# --- Set up the Streamlit page ---
st.set_page_config(layout="wide")
st.title("Restaurant Data Analysis Dashboard")
st.write("This dashboard analyzes a restaurant dataset to provide insights on cuisines, cities, ratings, and more.")

# --- Load the data ---
@st.cache_data
def load_data():
    df = pd.read_csv('/Users/palammysurareddy/Downloads/intership project 11:08:2025 /Dataset  - Dataset .csv')
    return df

df = load_data()

st.header("1. Dataset Overview")
st.write("Here is a sample of the data:")
st.dataframe(df.head())

# --- Analysis: Top Cuisines ---
st.header("2. Top Cuisines Analysis")
cuisine_column = 'Cuisines'
if cuisine_column in df.columns:
    st.subheader("Top 3 Cuisines and Their Percentages")
    cuisine_counts = df[cuisine_column].value_counts()
    top_cuisines = cuisine_counts.head(3)
    total_restaurants = len(df)
    top_cuisines_percentage = (top_cuisines / total_restaurants) * 100
    result = pd.DataFrame({
        'Cuisine': top_cuisines.index,
        'Count': top_cuisines.values,
        'Percentage': top_cuisines_percentage.round(2)
    })
    st.dataframe(result)
else:
    st.warning(f"The dataset does not contain a '{cuisine_column}' column.")

# --- Analysis: City Insights ---
st.header("3. City Insights")
city_column = "City"
rating_column = "Aggregate rating"
if city_column in df.columns and rating_column in df.columns:
    st.subheader("Distribution of Restaurants by City")
    city_counts = df[city_column].value_counts()
    top_city_by_count = city_counts.idxmax()
    top_city_count = city_counts.max()
    st.write(f"City with the highest number of restaurants: **{top_city_by_count}** ({top_city_count} restaurants)")
    
    avg_rating_by_city = df.groupby(city_column)[rating_column].mean()
    top_city_by_rating = avg_rating_by_city.idxmax()
    top_avg_rating = avg_rating_by_city.max()
    st.write(f"City with the highest average rating: **{top_city_by_rating}** ({top_avg_rating:.2f})")
    
    st.subheader("Average Rating by City")
    st.dataframe(avg_rating_by_city.round(2))
else:
    st.warning(f"The dataset does not contain '{city_column}' or '{rating_column}' columns.")

# --- Analysis: Price Range Distribution ---
st.header("4. Price Range Distribution")
price_column = "Price range"
if price_column in df.columns:
    price_counts = df[price_column].value_counts().sort_index()
    total_restaurants = len(df)
    price_percentages = (price_counts / total_restaurants) * 100
    st.subheader("Percentage of Restaurants in Each Price Range")
    st.dataframe(price_percentages.round(2))

    fig, ax = plt.subplots(figsize=(8,5))
    price_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title("Distribution of Price Ranges Among Restaurants")
    ax.set_xlabel("Price Range")
    ax.set_ylabel("Number of Restaurants")
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig)
else:
    st.warning(f"The dataset does not contain a '{price_column}' column.")

# --- Analysis: Online Delivery vs. Ratings ---
st.header("5. Online Delivery vs. Ratings")
online_delivery_col = "Has Online delivery"
rating_col = "Aggregate rating"
if online_delivery_col in df.columns and rating_col in df.columns:
    delivery_counts = df[online_delivery_col].value_counts()
    percentage_online_delivery = (delivery_counts.get("Yes", 0) / len(df)) * 100
    st.write(f"Percentage of restaurants with online delivery: **{percentage_online_delivery:.2f}%**")
    
    avg_rating_comparison = df.groupby(online_delivery_col)[rating_col].mean().round(2)
    st.subheader("Average Ratings (Online Delivery vs. No)")
    st.dataframe(avg_rating_comparison)
else:
    st.warning(f"The dataset does not contain '{online_delivery_col}' or '{rating_col}' columns.")

# --- Analysis: Restaurant Chains ---
st.header("6. Restaurant Chain Analysis")
name_col = "Restaurant Name"
votes_col = "Votes"
if name_col in df.columns and rating_col in df.columns and votes_col in df.columns:
    chain_counts = df[name_col].value_counts()
    restaurant_chains = chain_counts[chain_counts > 1]
    
    st.subheader("Ratings and Popularity of Restaurant Chains")
    chain_data = df[df[name_col].isin(restaurant_chains.index)]
    chain_analysis = chain_data.groupby(name_col).agg({
        rating_col: "mean",
        votes_col: "mean",
        name_col: "count"
    }).rename(columns={name_col: "Number of Outlets"})
    chain_analysis = chain_analysis.sort_values(by="Number of Outlets", ascending=False)
    st.dataframe(chain_analysis.round(2))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    chain_analysis[rating_col].plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title("Average Ratings of Restaurant Chains")
    ax.set_xlabel("Restaurant Chain")
    ax.set_ylabel("Average Rating")
    ax.tick_params(axis='x', rotation=75)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning(f"The dataset does not contain '{name_col}', '{rating_col}', or '{votes_col}' columns.")

# --- Analysis: Price Range vs. Services ---
st.header("7. Services Availability by Price Range")
price_col = "Price range"
delivery_col = "Has Online delivery"
booking_col = "Has Table booking"
if price_col in df.columns and delivery_col in df.columns and booking_col in df.columns:
    price_analysis = df.groupby(price_col).agg({
        delivery_col: lambda x: (x == "Yes").mean() * 100,
        booking_col: lambda x: (x == "Yes").mean() * 100
    }).rename(columns={
        delivery_col: "% Offering Online Delivery",
        booking_col: "% Offering Table Booking"
    })
    st.dataframe(price_analysis.round(2))

    fig, ax = plt.subplots(figsize=(8, 5))
    price_analysis.plot(kind='bar', ax=ax)
    ax.set_title("Services Availability by Price Range")
    ax.set_ylabel("Percentage of Restaurants")
    ax.set_xlabel("Price Range")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Service")
    ax.grid(axis='y')
    st.pyplot(fig)
else:
    st.warning(f"The dataset does not contain '{price_col}', '{delivery_col}', or '{booking_col}' columns.")

# --- Analysis: Detailed Rating and Votes ---
st.header("8. Detailed Rating and Votes Analysis")
votes_col = "Votes"
rating_col = "Aggregate rating"
if votes_col in df.columns and rating_col in df.columns:
    highest_votes = df.loc[df[votes_col].idxmax()]
    lowest_votes = df.loc[df[votes_col].idxmin()]
    correlation = df[[votes_col, rating_col]].corr().iloc[0, 1]
    
    st.subheader("Restaurant with Highest and Lowest Votes")
    st.write("Restaurant with highest votes:")
    st.dataframe(highest_votes)
    st.write("Restaurant with lowest votes:")
    st.dataframe(lowest_votes)
    
    st.write(f"**Correlation between number of votes and rating:** `{correlation:.2f}`")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df[votes_col], df[rating_col], alpha=0.5, color='yellow', edgecolors='black')
    ax.set_title("Votes vs. Rating")
    ax.set_xlabel("Number of Votes")
    ax.set_ylabel("Rating")
    ax.grid(True)
    st.pyplot(fig)
else:
    st.warning(f"The dataset does not contain '{votes_col}' or '{rating_col}' columns.")

# --- Analysis: Geospatial Analysis (Restaurant Locations) ---
st.header("9. Geospatial Analysis")
lat_col = "Latitude"
lon_col = "Longitude"
if lat_col in df.columns and lon_col in df.columns:
    df = df.dropna(subset=[lat_col, lon_col])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(df[lon_col], df[lat_col], alpha=0.5, s=10, color="blue", edgecolors='none')
    ax.set_title("Restaurant Locations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    st.pyplot(fig)
else:
    st.warning(f"The dataset does not contain '{lat_col}' or '{lon_col}' columns.")

# --- Analysis: Sentiment Analysis (Placeholder) ---
st.header("10. Sentiment Analysis")
# This section assumes a "Review Text" column, which is not in the original CSV.
# The original code adds a fake column, so we'll do the same.
review_col = "Review Text" # Original, but we'll use a new one
rating_col = "Aggregate rating"
try:
    if review_col not in df.columns:
        # Add a fake review column as in the original code
        st.info("Creating a placeholder 'Review' column for sentiment analysis.")
        sample_reviews = [
            "Great food and service!", "Average experience, nothing special.",
            "Will definitely come back again!", "Food was cold and service was slow.",
            "Excellent taste and friendly staff."
        ]
        df["Review"] = [random.choice(sample_reviews) for _ in range(len(df))]
        review_col = "Review" # Update the column name
        df = df.dropna(subset=[review_col])

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\\s]", "", text)
        return text

    df["cleaned_review"] = df[review_col].apply(clean_text)
    
    positive_words = ["good", "great", "amazing", "excellent", "delicious", "friendly", "perfect", "love", "nice", "tasty"]
    negative_words = ["bad", "poor", "worst", "terrible", "awful", "slow", "disappointing", "hate", "rude", "bland"]
    
    all_words = " ".join(df["cleaned_review"]).split()
    word_counts = Counter(all_words)
    
    positive_counts = {word: word_counts.get(word, 0) for word in positive_words}
    negative_counts = {word: word_counts.get(word, 0) for word in negative_words}
    
    st.subheader("Common Positive and Negative Keywords")
    st.write("Most Common Positive Keywords:", positive_counts)
    st.write("Most Common Negative Keywords:", negative_counts)
    
    df["review_length"] = df["cleaned_review"].apply(lambda x: len(x.split()))
    avg_length = df["review_length"].mean()
    st.write(f"Average Review Length: `{avg_length:.2f}` words")
    
    st.subheader("Review Length vs. Rating")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["review_length"], df[rating_col], alpha=0.5)
    ax.set_title("Review Length vs. Rating")
    ax.set_xlabel("Review Length (words)")
    ax.set_ylabel("Rating")
    ax.grid(True)
    st.pyplot(fig)

except KeyError as e:
    st.warning(f"Skipping Sentiment Analysis due to missing column: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred in sentiment analysis: {e}")