import streamlit as st
import os
import logging
import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
import tiktoken
from geopy.geocoders import Nominatim
import googlemaps
from streamlit_folium import st_folium
import folium
import google.generativeai as genai

# Additional imports for image processing
import torch
import torchvision.transforms as transforms
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API Keys
if not GROQ_API_KEY or not GOOGLE_PLACES_API_KEY or not GEMINI_API_KEY:
    st.error("API keys for Groq, Google Places, and/or Gemini are missing. Check your environment variables.")
    st.stop()

# Initialize APIs
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
gmaps = googlemaps.Client(key=GOOGLE_PLACES_API_KEY)

genai.configure(api_key=GEMINI_API_KEY)
model_name = "gemini-1.5-flash"
model = genai.GenerativeModel(model_name)

# Define llm.invoke to use Google's Generative AI
class GoogleLLMWrapper:
    def invoke(self, prompt):
        try:
            response = model.generate_content(prompt)
            if response and hasattr(response, "text"):  # Check if response is valid
                return {"content": response.text}  # Return the text content
            else:
                logging.error("Invalid response from Gemini model.")
                return None
        except Exception as e:
            logging.error(f"Error invoking Gemini model: {e}")
            return None

llm = GoogleLLMWrapper()

def count_tokens(text):
    """Count the number of tokens in the input text."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# Safe Summarization Function
def safe_summarize(content, token_limit=2000):
    """Summarize content to stay within token limits."""
    if count_tokens(content) > token_limit:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer(content, max_length=300, min_length=50, do_sample=False)[0]['summary_text']
    return content

# Extract Text from PDFs
def extract_text_from_pdfs(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text") or ""
        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {e}")
    return text

# Generate Embeddings
def generate_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = split_text_into_chunks(text)
    if not chunks:
        logging.error("No chunks generated for embedding.")
        return [], None
    embeddings = model.encode(chunks)
    return chunks, embeddings

# Create FAISS Index
def create_faiss_index(embeddings):
    if embeddings.size == 0:
        logging.error("No embeddings provided to create FAISS index.")
        return None

    dim = embeddings.shape[1]  # Dimension of the embeddings
    nlist = min(100, embeddings.shape[0])  # Ensure nlist <= number of training points

    # Create the FAISS index
    quantizer = faiss.IndexFlatL2(dim)  # Quantizer for IVF
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)  # Use nlist instead of a fixed value

    # Train the index
    if embeddings.shape[0] >= nlist:  # Ensure there are enough training points
        index.train(embeddings)
    else:
        logging.warning(f"Not enough training points ({embeddings.shape[0]}) for {nlist} clusters. Using a flat index instead.")
        index = faiss.IndexFlatL2(dim)  # Fallback to a flat index

    # Add embeddings to the index
    index.add(embeddings)
    return index

# Split Text into Chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

# Search for Relevant Content in PDFs
def get_top_chunks(query, index, pdf_chunks, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=top_k)
    top_chunks = [pdf_chunks[i] for i in I[0] if i < len(pdf_chunks)]
    return top_chunks

# Dynamic Prompt Builder
def build_dynamic_prompt(context, description, num_chunks):
    prompt_template = """
    "You are a medical expert. Analyze the following skin issue based on {num_chunks} retrieved medical references. Prioritize common skin conditions unless the symptoms strongly suggest a rare or severe disease."
    Context: {context}
    User Description: {description}
    Provide the top 5 possible diseases in JSON strictly JSON format:
    [
        {{"disease": "<Disease Name 1>", "score": <Possibility Score 1>}},
        ...
    ]
    No explanation or additional text should be provided. Only send JSON output.
    """
    human_prompt = HumanMessagePromptTemplate.from_template(prompt_template)
    return ChatPromptTemplate.from_messages([human_prompt])

# Generate Diagnostic Questions
def generate_questions_with_options(disease_list, llm, max_questions=10):
    """
    Generate diagnostic questions with options tailored to the predicted diseases using the LLM.
    """
    # Prepare input for the LLM
    prompt = f"""
    You are a medical expert. The following diseases are possible based on user input: {', '.join(disease_list)}.
    Generate up to {max_questions} diagnostic multiple-choice questions to help a person distinguish which disease they have among those in this list.
    Each question should have 3-4 options, and the options should help differentiate uniquely which disease the patient has.
    Questions should be based on user's experience with disease symptoms and not general knowledge questions on disease.
    Provide the questions and options in the following JSON format:
    [
        {{
            "question": "Question 1 text",
            "options": ["Option A", "Option B", "Option C"],
            "answer": "Option A"
        }},
        ...
    ]
    IMPORTANT:
    - Do not include any explanations, additional text, or notes.
    - Return only the JSON output.
    - Ensure the output is valid JSON and can be parsed by a JSON parser.
    """
    # Use the LLM to generate questions
    response = llm.invoke(prompt)
    questions_json = response["content"].strip() if response else ""

    logging.info(f"Questions JSON: {questions_json}")

    # Extract JSON content from the response
    try:
        # Find the start and end of the JSON content
        start = questions_json.find("[")
        end = questions_json.rfind("]") + 1

        if start == -1 or end == 0:
            logging.error("No JSON content found in response.")
            return []

        # Extract the JSON portion
        json_content = questions_json[start:end]

        # Parse the JSON
        questions = json.loads(json_content)

        # Validate the structure of the JSON
        if not isinstance(questions, list):
            raise ValueError("Expected a list of questions.")

        # Ensure each question has the required keys
        for question in questions:
            if not all(key in question for key in ["question", "options", "answer"]):
                raise ValueError("Invalid question format in JSON.")

        return questions[:max_questions]

    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Failed to parse questions JSON: {e}")
        return []

# Refine Disease Prediction
def refine_disease_prediction(disease_list, user_answers, llm):
    prompt = f"""
    You are a medical expert. Based on the user's symptoms, the following diseases were predicted: {', '.join(disease_list)}.
    The user has answered diagnostic questions as follows:
    {json.dumps(user_answers, indent=2)}
    Use this information to refine the predictions and provide the most likely disease.
    Output the final disease in this JSON format:
    {{
        "disease": "<Final Disease>",
        "justification": "<Reason based on answers>"
    }}
    Ensure the output is valid JSON without any additional text.
    """
    response = llm.invoke(prompt)
    refined_disease_str = response["content"].strip() if response else None
    # Attempt to parse JSON
    start = refined_disease_str.find("{") if refined_disease_str else -1
    end = refined_disease_str.rfind("}") + 1 if refined_disease_str else 0
    if start == -1 or end == 0:
        logging.error("No JSON content found in response.")
        return {"disease": "Unknown", "justification": "Unable to parse response."}
    try:
        refined_disease = json.loads(refined_disease_str[start:end])
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response: {e}")
        refined_disease = {"disease": "Unknown", "justification": "Unable to parse response."}
    return refined_disease

# Fetch Nearby Places
def fetch_nearby_places(place_type, user_coordinates, radius=2000):
    try:
        results = gmaps.places_nearby(location=user_coordinates, radius=radius, type=place_type).get("results", [])
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

# Display Map
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

# Actions Prompt
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

        Ensure the JSON response is well-formed and adheres strictly to the provided structure. Do not include any extra text outside the JSON object.
        """
    )

    # Format the prompt with the provided context, disease name, and severity
    formatted_prompt = prompt.format_prompt(context=context, disease_name=disease_name, severity=severity).to_string()

    return formatted_prompt

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

# Load the image model
IMAGE_MODEL_PATH = 'skin_disease_model.pth'

# Define the model architecture if needed (Uncomment and modify if using state_dict)
# from torchvision import models

# class SkinDiseaseModel(torch.nn.Module):
#     def __init__(self, num_classes=10):
#         super(SkinDiseaseModel, self).__init__()
#         self.model = models.resnet50(pretrained=False)
#         num_features = self.model.fc.in_features
#         self.model.fc = torch.nn.Linear(num_features, num_classes)  # Adjust num_classes

#     def forward(self, x):
#         return self.model(x)

# try:
#     image_model = SkinDiseaseModel(num_classes=10)  # Adjust num_classes accordingly
#     image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=torch.device('cpu')))
#     image_model.eval()
#     logging.info("Image model loaded successfully from state dictionary.")
# except Exception as e:
#     logging.error(f"Error loading image model: {e}")
#     st.error("Failed to load the image model. Please check the model file and architecture.")

# Alternatively, if loading the complete model
try:
    image_model = torch.load(IMAGE_MODEL_PATH, map_location=torch.device('cpu'))
    image_model.eval()
    logging.info("Image model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading image model: {e}")
    st.error("Failed to load the image model. Please check the model file.")

# Define image transformations (modify based on your model's training)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as per model's input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Mean values
                         [0.229, 0.224, 0.225])  # Std deviation
])

def map_disease_label(index):
    """
    Maps a class index to the corresponding disease name.
    Update this mapping based on your model's training labels.
    """
    disease_labels = {
        0: 'Acne',
        1: 'Eczema',
        2: 'Psoriasis',
        3: 'Rosacea',
        4: 'Vitiligo',
        5: 'Melanoma',
        6: 'Dermatitis',
        7: 'Hives',
        8: 'Warts',
        9: 'Other'
        # Add all necessary mappings
    }
    return disease_labels.get(index, 'Unknown Disease')

def predict_disease_from_image(image_file):
    try:
        image = Image.open(image_file).convert('RGB')
        image_tensor = image_transforms(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = image_model(image_tensor)
            # If the model outputs logits
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(probabilities, 1)
            predicted_disease = map_disease_label(predicted_idx.item())
            return predicted_disease
    except Exception as e:
        logging.error(f"Error during image prediction: {e}")
        return "Prediction Failed"

# Process image input if uploaded
if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.info("Processing image...")
    image_pred = predict_disease_from_image(uploaded_image)
    st.success(f"Image-based Prediction: {image_pred}")
    disease_predictions.append(image_pred)
else:
    st.warning("No image uploaded. Proceeding with text-based predictions only.")

try:
    # Extract and Process PDF Data
    pdf_paths = ["skin_diseases_guide1.pdf", "skin_diseases_guide2.pdf"]

    def clean_pdf_text(text):
        """Remove irrelevant sections like acknowledgments, prefaces, and exam questions."""
        filtered_text = [line.strip() for line in text.split("\n") if not any(keyword in line for keyword in ["ACKNOWLEDGEMENT", "PREFACE", "Table", "exam"])]
        return " ".join(filtered_text)

    # Extract text from PDFs
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
    query_context = {"context": context, "description": user_description, "num_chunks": num_chunks}
    formatted_prompt = prompt.format_prompt(**query_context).to_string()

    logging.info(f"Final prompt for LLM:\n{formatted_prompt}")
    response = llm.invoke(formatted_prompt)

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
#maskeen
        # Generate diagnostic questions based on combined predictions
        diagnostic_questions = generate_questions_with_options(disease_predictions, llm, max_questions=10)

        if diagnostic_questions:
            st.markdown("### Diagnostic Questions:")

            # Initialize session state for user answers if not already present
            if "user_answers" not in st.session_state:
                st.session_state.user_answers = {}

            for question in diagnostic_questions:
                if question["question"] not in st.session_state.user_answers:
                    st.session_state.user_answers[question["question"]] = st.selectbox(
                        question["question"],
                        question["options"],
                        key=question["question"]
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
        pdf_paths = ["remedies1.pdf", "remedies2.pdf"]
        pdf_text = clean_pdf_text(extract_text_from_pdfs(pdf_paths))
        pdf_chunks, pdf_embeddings = generate_embeddings(pdf_text)
        pdf_index = create_faiss_index(pdf_embeddings)

        # Query the PDF for remedies based on the disease name
        context = " ".join(get_top_chunks(disease_name, pdf_index, pdf_chunks))

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
            st.error("Invalid location. Please try entering a different location.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        st.error(f"An error occurred while processing the location: {e}")
