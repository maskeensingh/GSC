import googlemaps
import logging
from geopy.geocoders import Nominatim
import folium

def fetch_nearby_places(place_type, user_coordinates, gmaps_client, radius=2000):
    try:
        results = gmaps_client.places_nearby(location=user_coordinates, radius=radius, type=place_type).get("results", [])
        # Additional filtering for skin specialists
        if place_type == "skin specialist":
            keywords = ["dermatologist", "skin", "clinic", "hospital"]
            results = [place for place in results if any(keyword.lower() in place.get("name", "").lower() for keyword in keywords)]
        # Extract relevant information
        places = []
        for place in results:
            name = place.get("name", "Unknown")
            address = place.get("vicinity", "Address not available")
            rating = place.get("rating", "No rating available")
            location = place.get("geometry", {}).get("location", {})
            if location:
                places.append({
                    "name": name,
                    "address": address,
                    "rating": rating,
                    "location": (location.get("lat"), location.get("lng"))
                })
            if len(places) >= 5:
                break
        return places
    except Exception as e:
        logging.error(f"Google Places API Error: {e}")
        return []

def display_map(places, user_coordinates):
    m = folium.Map(location=user_coordinates, zoom_start=15)
    folium.Marker(
        location=user_coordinates, popup="Your Location", icon=folium.Icon(color="blue")
    ).add_to(m)
    for place in places:
        folium.Marker(
            location=place["location"],
            popup=f"{place['name']} - {place.get('rating', 'No rating')}⭐",
            icon=folium.Icon(color="green")
        ).add_to(m)
    # Save map to HTML string
    return m._repr_html_()

def get_coordinates(location_input):
    try:
        geolocator = Nominatim(user_agent="skingenie")
        user_location = geolocator.geocode(location_input)
        if user_location:
            return (user_location.latitude, user_location.longitude)
        else:
            return None
    except Exception as e:
        logging.error(f"Geocoding error: {e}")
        return None
