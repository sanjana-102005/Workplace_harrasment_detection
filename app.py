import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import pandas as pd


# ----------------------------
# TEXT CLEANING FUNCTION
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------------
# LOAD MODEL (CACHED)
# ----------------------------
@st.cache_resource
def load_model():
    model_path = "harrasment_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


# ----------------------------
# LABELS
# ----------------------------
labels = {
    0: "No Harassment",
    1: "Sexual Harassment",
    2: "Verbal Abuse",
    3: "Physical Threat"
}


# ----------------------------
# WORKPLACE CONTEXT KEYWORDS
# ----------------------------
context_keywords = [
    "manager","boss","supervisor","coworker","colleague",
    "employee","staff","office","team","work","company",
    "client","meeting","project","department","leader"
]


def is_workplace_related(text):
    text = text.lower()
    return any(word in text for word in context_keywords)


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict(text):

    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    probabilities = probs.tolist()[0]

    pred_class = torch.argmax(probs).item()
    confidence = torch.max(probs).item()

    context = is_workplace_related(text)

    return pred_class, confidence, probabilities, context


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("AI Workplace Harassment Detection")

st.markdown("""
This AI system detects:

• Sexual Harassment  
• Verbal Abuse  
• Physical Threats  

The model analyzes workplace complaints and predicts harassment type.
""")


# ----------------------------
# USER INPUT
# ----------------------------
user_input = st.text_area("Enter workplace complaint")


# ----------------------------
# EXAMPLE BUTTONS
# ----------------------------
st.subheader("Example Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Sexual Harassment"):
        user_input = "My manager keeps making sexual comments"

with col2:
    if st.button("Verbal Abuse"):
        user_input = "My boss insulted me in front of everyone"

with col3:
    if st.button("Physical Threat"):
        user_input = "My supervisor threatened to hit me"


# ----------------------------
# ANALYZE BUTTON
# ----------------------------
if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter text")

    else:

        pred_class, confidence, probabilities, context = predict(user_input)

        prediction = labels[pred_class]

        st.subheader("Prediction Result")

        if prediction == "No Harassment":
            st.success(prediction)

        else:
            st.error(prediction)


        st.write("Confidence:", round(confidence * 100, 2), "%")

        st.progress(int(confidence * 100))


        # ----------------------------
        # CONFIDENCE LEVEL
        # ----------------------------
        if confidence > 0.9:
            st.info("Confidence Level: High")

        elif confidence > 0.75:
            st.info("Confidence Level: Moderate")

        else:
            st.warning("Confidence Level: Low")


        # ----------------------------
        # CLASS PROBABILITIES
        # ----------------------------
        st.subheader("Class Probabilities")

        for i, label in labels.items():
            score = probabilities[i]
            st.write(label, ":", round(score * 100, 2), "%")
            st.progress(score)


        # ----------------------------
        # CONTEXT CHECK
        # ----------------------------
        if context:
            st.success("Workplace context detected")

        else:
            st.warning("Workplace context unclear")


        # ----------------------------
        # SAVE PREDICTION LOG
        # ----------------------------
        log = pd.DataFrame({
            "text":[user_input],
            "prediction":[prediction],
            "confidence":[confidence]
        })

        log.to_csv("prediction_log.csv", mode="a", header=False, index=False)


# ----------------------------
# SIDEBAR LIMITATIONS
# ----------------------------
st.sidebar.title("System Limitations")

st.sidebar.info("""
• The model may produce false positives or false negatives  
• Context detection is keyword-based  
• Predictions should not replace human judgement  
""")