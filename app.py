import streamlit as st
import os
import logging
import json
import faiss
from dotenv import load_dotenv
from PIL import Image
import torch
import torchvision.transforms as transforms

# Import utility modules
from utils.pdf_processing import (
    extract_text_from_pdfs,
    clean_pdf_text,
    split_text_into_chunks,
    generate_embeddings,
    create_faiss_index,
    get_top_chunks,
    safe_summarize
)
from utils.image_processing import (
    load_image_model,
    predict_disease_from_image
)
from utils.llm_integration import (
    initialize_llm,
    build_dynamic_prompt,
    generate_questions_with_options,
    refine_disease_prediction,
    actions_prompt
)
from utils.location_services import (
    fetch_nearby_places,
    display_map,
    get_coordinates
)

# Additional imports
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import googlemaps
import google.generativeai as genai
from geopy.geocoders import Nominatim

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables via Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GOOGLE_PLACES_API_KEY = st.secrets["GOOGLE_PLACES_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Validate API Keys
if not GROQ_API_KEY or not GOOGLE_PLACES_API_KEY or not GEMINI_API_KEY:
    st.error("API keys for Groq, Google Places, and/or Gemini are missing. Check your environment variables.")
    st.stop()

# Initialize APIs
llm = initialize_llm(GROQ_API_KEY, GEMINI_API_KEY)
gmaps = googlemaps.Client(key=GOOGLE_PLACES_API_KEY)

# Load the image model
IMAGE_MODEL_PATH = 'models/skin_disease_model.pth'

try:
    image_model = load_image_model(IMAGE_MODEL_PATH)
    st.info("Image model loaded successfully.")
except Exception as e:
    st.error("Failed to load the image model. Please check the model file and architecture.")
    st.stop()

# Define token counting function
def count_tokens(text):
    """Count the number of tokens in the input text."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# Streamlit Interface
st.title("Skin Disease Diagnosis Assistant")
st.markdown("""
    🧬 **Diagnose Skin Diseases**: Upload an image, provide a description of the skin problem, and receive guidance on possible diseases along with diagnostic questions for refinement.
""")

# User Inputs
uploaded_image = st.file_uploader("Upload an image of the skin issue (optional):", type=["jpg", "jpeg", "png"])
user_description = st.text_area("Describe your skin problem (e.g., 'Red rash with itchiness for 3 days'):")
location_input = st.text_input("Enter your location (e.g., 'New Delhi, India'):").strip()

# Early validation of description input
if not user_description or len(user_description.strip()) < 5:
    st.error("Please provide a detailed description of your skin problem.")
    st.stop()  # Stop further processing if input is invalid

# Initialize a list to hold disease predictions
disease_predictions = []

# Process image input if uploaded
if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.info("Processing image...")
    image_pred = predict_disease_from_image(image_model, uploaded_image)
    st.success(f"Image-based Prediction: {image_pred}")
    disease_predictions.append(image_pred)
else:
    st.warning("No image uploaded. Proceeding with text-based predictions only.")

try:
    # Extract and Process PDF Data
    pdf_paths = ["pdfs/skin_diseases_guide1.pdf", "pdfs/skin_diseases_guide2.pdf"]

    pdf_text = extract_text_from_pdfs(pdf_paths)
    if not pdf_text:
        st.error("Failed to extract text from provided PDF files.")
        st.stop()

    logging.info("PDF text extracted successfully.")
    pdf_text = clean_pdf_text(pdf_text)
    logging.info("PDF text cleaned successfully.")
    
    # Generate embeddings and create index
    pdf_chunks, pdf_embeddings = generate_embeddings(pdf_text)
    logging.info("Embeddings generated successfully.")
    pdf_index = create_faiss_index(pdf_embeddings)
    
    if not pdf_index:
        st.error("Failed to create FAISS index. Ensure embeddings were generated correctly.")
        st.stop()

    logging.info("FAISS index created successfully.")

    # Search for relevant content based on user description
    pdf_search_results = get_top_chunks(user_description, pdf_index, pdf_chunks, top_k=3)
    context = safe_summarize(" ".join(pdf_search_results), token_limit=3000)

    num_chunks = min(len(pdf_search_results), 5)
    prompt = build_dynamic_prompt(context, user_description, num_chunks)

    logging.info(f"Final prompt for LLM:\n{prompt}")
    response = llm.invoke(prompt)

    if not response or not response["content"].strip():
        st.error("Unable to process diagnosis. Please try again.")
        st.stop()

    raw_response = response["content"].strip()
    logging.info(f"Raw LLM response: {raw_response}")

    # Parsing the response
    try:
        diseases_with_scores = json.loads(raw_response)
        logging.info(f"LLM returned diseases with scores: {diseases_with_scores}")
    except json.JSONDecodeError:
        logging.error("Failed to parse JSON response.")
        st.error("Unable to process the diseases with scores. Please try again.")
        st.stop()

    if diseases_with_scores:
        st.markdown("### Possible Diseases with Scores:")
        st.json(diseases_with_scores)

        # Extract disease names from text-based predictions
        text_diseases = [disease["disease"] for disease in diseases_with_scores]
        disease_predictions.extend(text_diseases)  # Combine image and text predictions

        # Limit to top 6 diseases (1 image + 5 text)
        disease_predictions = disease_predictions[:6]

        st.markdown("### Combined Disease Predictions:")
        st.json(disease_predictions)

        # Generate diagnostic questions based on combined predictions
        diagnostic_questions = generate_questions_with_options(disease_predictions, llm, max_questions=10)

        if diagnostic_questions:
            st.markdown("### Diagnostic Questions:")

            # Initialize session state for user answers if not already present
            if "user_answers" not in st.session_state:
                st.session_state.user_answers = {}

            for idx, question in enumerate(diagnostic_questions, 1):
                if question["question"] not in st.session_state.user_answers:
                    st.session_state.user_answers[question["question"]] = st.selectbox(
                        question["question"],
                        question["options"],
                        key=f"question_{idx}"
                    )

            if st.button("Submit Answers"):
                st.success("Answers submitted successfully!")
                refined_prediction = refine_disease_prediction(disease_predictions, st.session_state.user_answers, llm)
                st.markdown("### Final Disease Prediction:")
                st.json(refined_prediction)

                if isinstance(refined_prediction, dict) and "disease" in refined_prediction:
                    disease_name = refined_prediction["disease"]
                    st.info(f"Stored final disease: {disease_name}")
                else:
                    st.warning("Could not extract a valid disease name from the refined prediction.")
                    disease_name = ""
        else:
            st.warning("No diagnostic questions generated. Please try again with more detailed input.")
    else:
        st.warning("No diseases were identified. Please try again with more detailed input.")
except Exception as e:
    logging.error(f"Error occurred: {e}")
    st.error(f"An error occurred while processing your request: {e}")
    st.stop()

# Additional handling for disease remedies and location if disease_name is found
if 'disease_name' in locals() and disease_name:
    try:
        # Fetch remedies related to the disease
        remedies_pdf_paths = ["pdfs/remedies1.pdf", "pdfs/remedies2.pdf"]
        remedies_text = extract_text_from_pdfs(remedies_pdf_paths)
        remedies_text = clean_pdf_text(remedies_text)
        remedies_chunks, remedies_embeddings = generate_embeddings(remedies_text)
        remedies_index = create_faiss_index(remedies_embeddings)

        # Query the PDF for remedies based on the disease name
        context = " ".join(get_top_chunks(disease_name, remedies_index, remedies_chunks))

        # Severity assessment
        severity_prompt = f"""
        Classify the disease {disease_name} in terms of its effects as one of the following: mild, moderate, or severe. 
        Only return one string as output: "mild", "moderate", or "severe". 
        Do not provide any explanations or additional text.
        """
        try:
            severity_response = llm.invoke(severity_prompt)
            severity = severity_response["content"].strip().lower() if severity_response else None
            logging.info(f"Severity returned: {severity}")

            if severity not in ["mild", "moderate", "severe"]:
                raise ValueError(f"Invalid severity level returned: {severity}")
        except Exception as e:
            logging.error(f"Error during severity assessment: {e}")
            st.error("An error occurred while assessing severity. Please try again.")
            severity = None

        if severity:
            # Generate the prompt for actionable advice
            prompt = actions_prompt(context, disease_name, severity)
            response = llm.invoke(prompt)
            logging.info(f"Model response: {response['content']}")  # Log the raw response

            # Attempt to parse the JSON from the model's response
            try:
                # Extract content between "```json" and "```" if delimiters are used
                if "```json" in response["content"]:
                    json_str = response["content"].split("```json")[1].strip().rstrip("```").strip()
                elif response["content"].strip().startswith("{") and response["content"].strip().endswith("}"):
                    json_str = response["content"].strip()
                else:
                    raise ValueError("Invalid JSON format in response.")

                # Parse the JSON
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
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        st.error(f"An error occurred while processing remedies: {e}")

# Handling user location and nearby services
if location_input:
    try:
        user_coordinates = get_coordinates(location_input)

        if user_coordinates:
            nearby_places_pharmacy = fetch_nearby_places("pharmacy", user_coordinates, gmaps)
            nearby_places_skin_specialist = fetch_nearby_places("skin specialist", user_coordinates, gmaps)

            if nearby_places_pharmacy:
                st.subheader("Nearby Pharmacies:")
                for place in nearby_places_pharmacy:
                    st.write(f"**{place['name']}** - {place['address']} ({place['rating']}⭐)")

            if nearby_places_skin_specialist:
                st.subheader("Nearby Skin Specialists:")
                for place in nearby_places_skin_specialist:
                    st.write(f"**{place['name']}** - {place['address']} ({place['rating']}⭐)")

            if nearby_places_pharmacy or nearby_places_skin_specialist:
                map_html = display_map(nearby_places_pharmacy + nearby_places_skin_specialist, user_coordinates)
                st.components.v1.html(map_html, width=700, height=500)
            else:
                st.warning("No nearby locations found.")
        else:
            st.error("Invalid location. Please try entering a different location.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        st.error(f"An error occurred while processing the location: {e}")
