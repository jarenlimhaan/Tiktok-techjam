import requests
from config.env import get_app_configs

API_KEY = get_app_configs().GOOGLE_PLACES_API_KEY

def get_reviews(query):
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,      
        "inputtype": "textquery",
        "fields": "place_id",
        "key": API_KEY
    }
    place_id = requests.get(url, params=params).json()["candidates"][0]["place_id"]

    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,rating,user_ratings_total,reviews",
        "reviews_sort": "newest",   
        "key": API_KEY
    }
    details = requests.get(url, params=params).json()["result"]
    return details.get("reviews", [])