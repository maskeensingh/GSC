import logging
import json
from langchain_groq import ChatGroq
import google.generativeai as genai

class GoogleLLMWrapper:
    def __init__(self, gemini_api_key, model_name="gemini-1.5-flash"):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)

    def invoke(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            if response and hasattr(response, "text"):
                return {"content": response.text}
            else:
                logging.error("Invalid response from Gemini model.")
                return None
        except Exception as e:
            logging.error(f"Error invoking Gemini model: {e}")
            return None

def initialize_llm(groq_api_key, gemini_api_key):
    # Initialize ChatGroq if needed, or directly use GoogleLLMWrapper
    llm = GoogleLLMWrapper(gemini_api_key)
    return llm

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
    return prompt_template.format(context=context, description=description, num_chunks=num_chunks)

def generate_questions_with_options(disease_list, llm, max_questions=10):
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
    response = llm.invoke(prompt)
    questions_json = response["content"].strip() if response else ""

    logging.info(f"Questions JSON: {questions_json}")

    try:
        questions = json.loads(questions_json)
        if not isinstance(questions, list):
            raise ValueError("Expected a list of questions.")
        for question in questions:
            if not all(key in question for key in ["question", "options", "answer"]):
                raise ValueError("Invalid question format in JSON.")
        return questions[:max_questions]
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Failed to parse questions JSON: {e}")
        return []

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

def actions_prompt(context, disease_name, severity):
    prompt = f"""
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
    return prompt
