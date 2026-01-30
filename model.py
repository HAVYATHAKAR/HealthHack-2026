
SPECIALTY_MAP = {
    "fungal infection": "Dermatology",
    "allergy": "Immunology",
    "gerd": "Gastroenterology",
    "chronic cholestasis": "Gastroenterology",
    "drug reaction": "Dermatology",
    "peptic ulcer disease": "Gastroenterology",
    "aids": "Infectious Disease",
    "diabetes": "Endocrinology",
    "gastroenteritis": "Gastroenterology",
    "bronchial asthma": "Pulmonology",
    "hypertension": "Cardiology",
    "migraine": "Neurology",
    "cervical spondylosis": "Orthopedics",
    "paralysis (brain hemorrhage)": "Neurology",
    "jaundice": "Hepatology",
    "malaria": "Infectious Disease",
    "chicken pox": "Infectious Disease",
    "dengue": "Infectious Disease",
    "typhoid": "Infectious Disease",
    "hepatitis a": "Hepatology",
    "hepatitis b": "Hepatology",
    "hepatitis c": "Hepatology",
    "hepatitis d": "Hepatology",
    "hepatitis e": "Hepatology",
    "alcoholic hepatitis": "Hepatology",
    "tuberculosis": "Pulmonology",
    "pneumonia": "Pulmonology",
    "common cold": "General Medicine",
    "influenza": "General Medicine",
    "urinary tract infection": "Urology",
    "varicose veins": "Vascular Surgery",
    "hypothyroidism": "Endocrinology",
    "hyperthyroidism": "Endocrinology",
    "arthritis": "Rheumatology",
    "osteoporosis": "Orthopedics",
    "heart attack": "Cardiology",
    "psoriasis": "Dermatology"
}

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load Data
df = pd.read_csv("choongqianzheng_disease_symptoms.csv")

# Map disease → specialty
df["specialty"] = df["disease"].map(SPECIALTY_MAP)

# Drop unmapped diseases (very few)
df = df.dropna(subset=["specialty"])

# Display first few rows
df.head()

# Prepare Data for Vectorization
X = df["symptoms"]
y = df["specialty"]

# FIX 1: Added token_pattern=None to remove the UserWarning
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x.split("|"),
    token_pattern=None,
    lowercase=False
)

X_vec = vectorizer.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))

# FIX 2: Added zero_division=0 to remove the UndefinedMetricWarning
print(classification_report(y_test, y_pred, zero_division=0))

# Prediction Function
def predict_specialty(symptom_list):
    """
    symptom_list: list of strings
    example: ["itching", "skin_rash", "nodal_skin_eruptions"]
    """
    symptom_text = "|".join(symptom_list)
    vec = vectorizer.transform([symptom_text])
    return model.predict(vec)[0]

# Example usage
# print("Predicted Specialty:", predict_specialty(["itching", "skin_rash", "nodal_skin_eruptions"]))
print("Predicted Specialty:", predict_specialty(["skin_rash","small_dents_in_nails","bladder_discomfort","pus_filled_pimples","anxiety","muscle_weakness"]))
