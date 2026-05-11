import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

df=pd.read_csv('/Users/palammysurareddy/Downloads/intership project 11:08:2025 /Dataset  - Dataset .csv')

df

print(df.columns)


print("Columns in dataset:", df.columns)


cuisine_column = 'Cuisines'
cuisine_column

if cuisine_column not in df.columns:
    raise ValueError(f"The dataset does not contain a '{cuisine_column}' column. Please check column names.")


cuisine_counts = df[cuisine_column].value_counts()
cuisine_counts

top_cuisines = cuisine_counts.head(3)
top_cuisines

top_cuisines

total_restaurants = len(df)
top_cuisines_percentage = (top_cuisines / total_restaurants) * 100
top_cuisines_percentage

result = pd.DataFrame({
    'Cuisine': top_cuisines.index,
    'Count': top_cuisines.values,
    'Percentage': top_cuisines_percentage.round(2)
})
result

print("Top 3 Cuisines and Their Percentages:")
print(result)


city_column = "City"  # Change if your dataset uses a different name
rating_column = "Aggregate rating" 

for col in [city_column, rating_column]:
    if col not in df.columns:
        raise ValueError(f"The dataset does not contain a '{col}' column.")

city_counts = df[city_column].value_counts()
top_city_by_count = city_counts.idxmax()
top_city_count = city_counts.max()

avg_rating_by_city = df.groupby(city_column)[rating_column].mean()
avg_rating_by_city

top_city_by_rating = avg_rating_by_city.idxmax()
top_avg_rating = avg_rating_by_city.max()
top_city_by_rating

top_avg_rating

print(f"City with the highest number of restaurants: {top_city_by_count} ({top_city_count} restaurants)")
print("\nAverage rating for restaurants in each city:")
print(avg_rating_by_city.round(2))
print(f"\nCity with the highest average rating: {top_city_by_rating} ({top_avg_rating:.2f})")

price_column = "Price range"
price_column

if price_column not in df.columns:
    raise ValueError(f"The dataset does not contain a '{price_column}' column.")

price_counts = df[price_column].value_counts().sort_index()
price_counts

total_restaurants = len(df)
price_percentages = (price_counts / total_restaurants) * 100
total_restaurants
price_percentages

price_percentages

print("Percentage of restaurants in each price range category:")
print(price_percentages.round(2))

plt.figure(figsize=(8,5))
price_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Distribution of Price Ranges Among Restaurants")
plt.xlabel("Price Range")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=0)
plt.show()

online_delivery_col = "Has Online delivery"   # Change if your dataset uses another name
rating_col = "Aggregate rating"
online_delivery_col
rating_col

for col in [online_delivery_col, rating_col]:
    if col not in df.columns:
        raise ValueError(f"The dataset does not contain a '{col}' column.")

delivery_counts = df[online_delivery_col].value_counts()
total_restaurants = len(df)
percentage_online_delivery = (delivery_counts.get("Yes", 0) / total_restaurants) * 100
delivery_counts
total_restaurants

delivery_counts

total_restaurants

percentage_online_delivery

avg_rating_comparison = df.groupby(online_delivery_col)[rating_col].mean().round(2)

# Output results
print(f"Percentage of restaurants with online delivery: {percentage_online_delivery:.2f}%")
print("\nAverage Ratings (Online Delivery vs No):")
print(avg_rating_comparison)

rating_col = "Aggregate rating"  # Change if needed
votes_col = "Votes" 
rating_col
votes_col

for col in [rating_col, votes_col]:
    if col not in df.columns:
        raise ValueError(f"The dataset does not contain a '{col}' column.")

rating_counts = df[rating_col].value_counts().sort_index()
rating_counts

most_common_rating = rating_counts.idxmax()
most_common_count = rating_counts.max()
most_common_count
most_common_rating

most_common_rating

avg_votes = df[votes_col].mean()
avg_votes

print("Distribution of Aggregate Ratings:")
print(rating_counts)
print(f"\nMost common rating: {most_common_rating} (count: {most_common_count})")
print(f"\nAverage number of votes per restaurant: {avg_votes:.2f}")

cuisine_col = "Cuisines"         
rating_col = "Aggregate rating"
cuisine_col,rating_col

for col in [cuisine_col, rating_col]:
    if col not in df.columns:
        raise ValueError(f"The dataset does not contain a '{col}' column.")

cuisine_combinations = df[cuisine_col].value_counts().head(10)
cuisine_combinations

avg_rating_by_combo = df.groupby(cuisine_col)[rating_col].mean().sort_values(ascending=False)
avg_rating_by_combo

print("Top 10 Most Common Cuisine Combinations:")
print(cuisine_combinations)
print("\nAverage Ratings for Cuisine Combinations:")
print(avg_rating_by_combo.round(2))

lat_col = "Latitude"   
lon_col = "Longitude"
lat_col,lon_col

for col in [lat_col, lon_col]:
    if col not in df.columns:
        raise ValueError(f"The dataset does not contain a '{col}' column.")

df = df.dropna(subset=[lat_col, lon_col])
df

plt.figure(figsize=(12, 5))
plt.scatter(df[lon_col], df[lat_col], alpha=0.5, s=10, color="blue", edgecolors='none')
plt.title("Restaurant Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

name_col = "Restaurant Name"     # Change if different
rating_col = "Aggregate rating"  # Change if different
votes_col = "Votes"
name_col,rating_col,votes_col

for col in [name_col, rating_col, votes_col]:
    if col not in df.columns:
        raise ValueError(f"The dataset does not contain a '{col}' column.")

chain_counts = df[name_col].value_counts()
restaurant_chains = chain_counts[chain_counts > 1]
chain_counts,restaurant_chains


chain_data = df[df[name_col].isin(restaurant_chains.index)]
chain_analysis = chain_data.groupby(name_col).agg({
    rating_col: "mean",
    votes_col: "mean",
    name_col: "count"
}).rename(columns={name_col: "Number of Outlets"})
chain_data

chain_analysis

rating_col

votes_col

chain_analysis = chain_analysis.sort_values(by="Number of Outlets", ascending=False)
chain_analysis

print("Restaurant Chains Found:")
print(restaurant_chains)
print("\nRatings and Popularity of Restaurant Chains:")
print(chain_analysis.round(2))

plt.figure(figsize=(10, 5))
chain_analysis[rating_col].plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Average Ratings of Restaurant Chains")
plt.xlabel("Restaurant Chain")
plt.ylabel("Average Rating")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

import re 
from collections import Counter

review_col = "Review Text"      
rating_col = "Aggregate rating"

import random

sample_reviews = [
    "Great food and service!",
    "Average experience, nothing special.",
    "Will definitely come back again!",
    "Food was cold and service was slow.",
    "Excellent taste and friendly staff."
]

# Add a fake review column
df["Review"] = [random.choice(sample_reviews) for _ in range(len(df))]

review_col = "Review"
rating_col = "Aggregate rating"


print(df.columns)


for col in [review_col, rating_col]:
    if col not in df.columns:
        raise ValueError(f"The dataset does not contain a '{col}' column.")

df = df.dropna(subset=[review_col])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
    return text

df["cleaned_review"] = df[review_col].apply(clean_text)
df

positive_words = ["good", "great", "amazing", "excellent", "delicious", "friendly", "perfect", "love", "nice", "tasty"]
negative_words = ["bad", "poor", "worst", "terrible", "awful", "slow", "disappointing", "hate", "rude", "bland"]
negative_words,positive_words

all_words = " ".join(df["cleaned_review"]).split()
word_counts = Counter(all_words)
all_words,word_counts

positive_counts = {word: word_counts[word] for word in positive_words if word in word_counts}
negative_counts = {word: word_counts[word] for word in negative_words if word in word_counts}
positive_counts

negative_counts

df["review_length"] = df["cleaned_review"].apply(lambda x: len(x.split()))
avg_length = df["review_length"].mean()
avg_length

plt.figure(figsize=(8, 5))
plt.scatter(df["review_length"], df[rating_col], alpha=0.5)
plt.title("Review Length vs. Rating")
plt.xlabel("Review Length (words)")
plt.ylabel("Rating")
plt.grid(True)
plt.show()

print("Most Common Positive Keywords:", positive_counts)
print("Most Common Negative Keywords:", negative_counts)
print(f"Average Review Length: {avg_length:.2f} words")


name_col = "Restaurant Name"     
votes_col = "Votes"               
rating_col = "Aggregate rating"

name_col

for col in [name_col, votes_col, rating_col]:
    if col not in df.columns:
        raise ValueError(f"The dataset does not contain a '{col}' column.")

highest_votes = df.loc[df[votes_col].idxmax()]
lowest_votes = df.loc[df[votes_col].idxmin()]
highest_votes

lowest_votes

correlation = df[[votes_col, rating_col]].corr().iloc[0, 1]
correlation

plt.figure(figsize=(10, 5))
plt.scatter(df[votes_col], df[rating_col], alpha=0.5, color='yellow', edgecolors='black')
plt.title("Votes vs Rating")
plt.xlabel("Number of Votes")
plt.ylabel("Rating")
plt.grid(True)
plt.show()

print("Restaurant with highest votes:")
print(highest_votes, "\n")
print("Restaurant with lowest votes:")
print(lowest_votes, "\n")
print(f"Correlation between number of votes and rating: {correlation:.2f}")

price_col = "Price range"           
delivery_col = "Has Online delivery"
booking_col = "Has Table booking"

price_col

for col in [price_col, delivery_col, booking_col]:
    if col not in df.columns:
        raise ValueError(f"The dataset does not contain a '{col}' column.")

price_analysis = df.groupby(price_col).agg({
    delivery_col: lambda x: (x == "Yes").mean() * 100,
    booking_col: lambda x: (x == "Yes").mean() * 100
}).rename(columns={
    delivery_col: "% Offering Online Delivery",
    booking_col: "% Offering Table Booking"
})

price_analysis

print(price_analysis.round(2))

price_analysis.plot(kind='bar', figsize=(8, 5))
plt.title("Services Availability by Price Range")
plt.ylabel("Percentage of Restaurants")
plt.xlabel("Price Range")
plt.xticks(rotation=0)
plt.legend(title="Service")
plt.grid(axis='y')
plt.show()


