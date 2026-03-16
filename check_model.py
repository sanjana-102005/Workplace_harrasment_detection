import os

print("Script started")

model_path = r"C:\Users\HP\Downloads\harrasment_detection_project\harrasment_model"

print("Checking path:", model_path)
print("Exists:", os.path.exists(model_path))

print("Files inside:")
print(os.listdir(model_path))