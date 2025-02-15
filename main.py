import streamlit as st
import os
import logging
import json
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tiktoken
from geopy.geocoders import Nominatim
import googlemaps
from streamlit_folium import st_folium
import folium

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# Validate API Keys
if not GROQ_API_KEY or not GOOGLE_PLACES_API_KEY:
    raise ValueError("API keys for Groq and/or Google Places are missing. Check your environment variables.")

# Initialize APIs
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
gmaps = googlemaps.Client(key=GOOGLE_PLACES_API_KEY)

# Helper Functions
def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def extract_text_from_pdfs(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text += " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            logging.error(f"Error reading {pdf_path}: {e}")
    return text

def clean_pdf_text(text):
    filtered_lines = [
        line.strip() for line in text.split("\n") 
        if not any(keyword in line for keyword in ["ACKNOWLEDGEMENT", "PREFACE", "Table", "exam"])
    ]
    return " ".join(filtered_lines)

def generate_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = split_text_into_chunks(text)
    embeddings = model.encode(chunks)
    return chunks, embeddings

def create_faiss_index(embeddings):
    if embeddings.size == 0:
        logging.error("No embeddings provided to create FAISS index.")
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def get_top_chunks(query, index, pdf_chunks, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=top_k)
    return [pdf_chunks[i] for i in I[0]] if len(I[0]) > 0 else []

def fetch_nearby_places(place_type, user_coordinates, radius=2000):
    try:
        results = gmaps.places_nearby(location=user_coordinates, radius=radius, type=place_type).get("results", [])
        
        # Filter results based on keywords for skin specialists
        if place_type == "skin specialist":
            keywords = ["dermatologist", "skin", "clinic","hospital"]
            results = [
                place for place in results
                if any(keyword.lower() in place.get("name", "").lower() for keyword in keywords)
            ]
        
        return [
            {
                "name": place.get("name", "Unknown"),
                "address": place.get("vicinity", "Address not available"),
                "rating": place.get("rating", "No rating available"),
                "location": (
                    place.get("geometry", {}).get("location", {}).get("lat"),
                    place.get("geometry", {}).get("location", {}).get("lng")
                ),
            }
            for place in results
        ][:5]
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
            popup=f"{place['name']} - {place.get('rating', 'No rating')}\u2b50",
            icon=folium.Icon(color="green")
        ).add_to(m)
    st_folium(m, width=700, height=500)

def actions_prompt(context, disease_name, severity):
    # Define the prompt template with dynamic placeholders
    prompt = ChatPromptTemplate.from_template(
        """
        You are a medical expert specializing in providing actionable advice to empower individuals, especially those with limited resources, to make informed decisions about their health. 

        Disease: {disease_name}
        Severity: {severity}
        
        Instructions:
        1. If the severity is **severe**, explain why the condition is critical, emphasize the urgency of consulting a doctor or specialist, and provide immediate actions or precautions to prevent worsening.
        2. If the severity is **moderate**, suggest affordable over-the-counter medications (ointments, creams, tablets, etc.), simple home remedies, and actions to manage symptoms effectively.
        3. Always include:
            - A list of common and affordable home remedies.
            - Low-cost, accessible medicines, if applicable.
            - Preventive measures to avoid further complications.
            - Necessary actions, dos and don’ts, and recovery tips.

        Context: {context}

        Output the response strictly in the following JSON format:
        ```json
        {{
            "severity": "{severity}",
            "answer": {{
                "description": "<Brief explanation of the condition>",
                "immediate_actions": [
                    "List of actions to take immediately for relief and to prevent worsening."
                ],
                "home_remedies": [
                    "Affordable, accessible remedies for symptom relief."
                ],
                "preventive_measures": [
                    "Steps to prevent the condition from worsening."
                ],
                "medications": [
                    "Over-the-counter medications available at most pharmacies."
                ],
                "dos_and_donts": [
                    "Helpful actions to take.",
                    "Actions to avoid for recovery."
                ]
            }}
        }}
        ```

        Example Output:
        ```json
        {{
            "severity": "severe",
            "answer": {{
                "description": "Severe eczema can lead to significant discomfort, sleep disturbance, and impact daily activities. It is essential to seek medical attention from a dermatologist or healthcare provider.",
                "immediate_actions": [
                    "Apply a topical corticosteroid or immunomodulator cream.",
                    "Use cool compresses or wet wraps.",
                    "Avoid scratching to prevent further irritation."
                ],
                "home_remedies": [
                    "Oatmeal baths for soothing.",
                    "Coconut oil to moisturize and calm the skin."
                ],
                "preventive_measures": [
                    "Identify and avoid allergens or irritants.",
                    "Keep skin moisturized with fragrance-free products."
                ],
                "medications": [
                    "Hydrocortisone cream for inflammation.",
                    "Diphenhydramine for itching relief."
                ],
                "dos_and_donts": [
                    "Do: Keep the skin hydrated.",
                    "Don’t: Use harsh soaps or detergents."
                ]
            }}
        }}
        ```

        Ensure the JSON response is well-formed and adheres strictly to the provided structure. Do not include any extra text outside the JSON object.
        """
    )



    # Format the prompt with the provided context, disease name, and severity
    formatted_prompt = prompt.format_prompt(context=context, disease_name=disease_name, severity=severity).to_string()

    return formatted_prompt

# Streamlit Interface
st.title("SkinGenie - Disease Remedy Finder")
st.markdown("🧬 Enter a skin disease and get remedies, nearby pharmacies, or dermatologists based on severity.")

# Inputs
disease_name = st.text_input("Enter the disease name (e.g., eczema, psoriasis):").strip()
location_input = st.text_input("Enter your location (e.g., 'New Delhi, India'):").strip()

if disease_name:
    try:
        # PDF Data
        pdf_paths = ["remedies1.pdf", "remedies2.pdf"]
        pdf_text = clean_pdf_text(extract_text_from_pdfs(pdf_paths))
        pdf_chunks, pdf_embeddings = generate_embeddings(pdf_text)
        pdf_index = create_faiss_index(pdf_embeddings)

        # Query PDF
        context = " ".join(get_top_chunks(disease_name, pdf_index, pdf_chunks))

        # Severity Assessment
        severity_prompt = f"""
        Classify the disease {disease_name} in terms of its effects as one of the following: mild, moderate, or severe. 
        Only return one string as output: "mild", "moderate", or "severe". 
        Do not provide any explanations or additional text.
        """
        try:
            severity = llm.invoke(severity_prompt).content.strip().lower()
            logging.info(f"Severity returned: {severity}")

            if severity not in ["mild", "moderate", "severe"]:
                raise ValueError(f"Invalid severity level returned: {severity}")
        except Exception as e:
            logging.error(f"Error during severity assessment: {e}")
            st.error("An error occurred while assessing severity. Please try again.")
            severity = None

        # Remedies Generation
        if severity:
            prompt = actions_prompt(context, disease_name, severity)
            response = llm.invoke(prompt)
            logging.info(f"Model response: {response.content}")  # Log the raw response

            # Attempt to parse the JSON from the model's response
            try:
                # Extract content between "```json" and "```" if delimiters are used
                if "```json" in response.content:
                    json_str = response.content.split("```json")[1].strip().rstrip("```").strip()
                elif response.content.strip().startswith("{") and response.content.strip().endswith("}"):
                    json_str = response.content.strip()
                else:
                    raise ValueError("Invalid JSON format in response.")
                remedies = json.loads(json_str)
            except Exception as e:
                logging.error(f"Error parsing JSON from response: {e}")
                st.error("An error occurred while parsing the recommendation. Please try again.")
                remedies = None

            # Display Remedies
            if remedies:
                st.subheader("Answer:")
                st.json(remedies)
            else:
                st.warning("No recommendations available at this time.")

        # Location Processing - Add map functionality if location is provided
        if location_input:
            geolocator = Nominatim(user_agent="skingenie")
            user_location = geolocator.geocode(location_input)
            if user_location:
                user_coordinates = (user_location.latitude, user_location.longitude)
                nearby_places_pharmacy = fetch_nearby_places("pharmacy", user_coordinates)
                nearby_places_skin_specialist = fetch_nearby_places("skin specialist", user_coordinates)
                
                if nearby_places_pharmacy:
                    st.subheader("Nearby Pharmacies:")
                    for place in nearby_places_pharmacy:
                        st.write(f"**{place['name']}** - {place['address']} ({place['rating']}⭐)")
                
                if nearby_places_skin_specialist:
                    st.subheader("Nearby Skin Specialists:")
                    for place in nearby_places_skin_specialist:
                        st.write(f"**{place['name']}** - {place['address']} ({place['rating']}⭐)")
                
                if nearby_places_pharmacy or nearby_places_skin_specialist:
                    display_map(nearby_places_pharmacy + nearby_places_skin_specialist, user_coordinates)
                else:
                    st.warning("No nearby locations found.")
            else:
                st.error("Invalid location.")
    except Exception as e:
        logging.error(f"Error: {e}")
        st.error("An error occurred. Please try again.")