import requests

API_KEY = "AIzaSyBt6cvBp1cBccsh-nvmnpM8Uz2_nKsgJHM"

def get_place_id(query):
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,             # e.g., "Mount Batur Sunrise Point"
        "inputtype": "textquery",
        "fields": "place_id",
        "key": API_KEY
    }
    return requests.get(url, params=params).json()["candidates"][0]["place_id"]

def get_reviews(place_id):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,rating,user_ratings_total,reviews",
        "reviews_sort": "newest",   # or "most_relevant"
        "key": API_KEY
    }
    return requests.get(url, params=params).json()["result"]

place_id = get_place_id("Mount Batur, Bali")
details = get_reviews(place_id)
for r in details.get("reviews", []):
    print(r["author_name"], r["rating"], r["relative_time_description"], r["text"])
