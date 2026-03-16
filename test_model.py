import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "harrasment_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

labels = {
    0: "No Harassment",
    1: "Sexual Harassment",
    2: "Verbal Abuse",
    3: "Physical Threat"
}

context_keywords = [
    "manager","boss","supervisor","coworker","colleague",
    "employee","staff","office","team","work","company",
    "client","meeting","project","department","leader"
]

def is_workplace_related(text):
    text = text.lower()
    return any(word in text for word in context_keywords)

def predict(text):

    if not is_workplace_related(text):
        print("\nText:", text)
        print("Prediction: Not able to detect (not workplace related)")
        return

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    pred_class = torch.argmax(probs).item()
    confidence = torch.max(probs).item()

    threshold = 0.70

    print("\nText:", text)

    if confidence < threshold:
        print("Prediction: Not able to detect")
        print("Confidence:", round(confidence*100,2), "%")
    else:
        print("Prediction:", labels[pred_class])
        print("Confidence:", round(confidence*100,2), "%")


while True:

    user_input = input("i love pizza ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    predict(user_input)