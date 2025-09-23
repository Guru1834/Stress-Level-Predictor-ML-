import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv("StressLevelDataset.csv")

features = [
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship", "future_career_concerns",
    "social_support", "peer_pressure", "extracurricular_activities", "bullying"
]

X = data[features]
y = data["stress_level"]   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
svm_model.fit(X_train, y_train)

feature_ranges = {col: (data[col].min(), data[col].max()) for col in features}

print("\nEnter Student Data:")
student_input = {}

feature_descriptions = {
    "anxiety_level": "anxiety level",
    "self_esteem": "self-esteem",
    "mental_health_history": "mental health history (0=no, 1=yes)",
    "depression": "depression level",
    "headache": "headache frequency",
    "blood_pressure": "blood pressure (1=low, 2=normal, 3=high)",
    "sleep_quality": "sleep quality",
    "breathing_problem": "breathing problems",
    "noise_level": "noise level in environment",
    "living_conditions": "living conditions",
    "safety": "feeling of safety",
    "basic_needs": "basic needs satisfaction",
    "academic_performance": "academic performance",
    "study_load": "study load",
    "teacher_student_relationship": "teacher-student relationship",
    "future_career_concerns": "future career concerns",
    "social_support": "social support",
    "peer_pressure": "peer pressure",
    "extracurricular_activities": "extracurricular activities",
    "bullying": "bullying experience"
}

for feature in features:
    min_val, max_val = feature_ranges[feature]
    description = feature_descriptions[feature]
    while True:
        try:
            value = int(input(f"Enter {description} (range {min_val}-{max_val}): "))
            if value < min_val or value > max_val:
                print(f"Please enter an integer within the range {min_val}-{max_val}.")
            else:
                student_input[feature] = value
                break
        except ValueError:
            print("Invalid input. Please enter an integer.")

student_df = pd.DataFrame([student_input])

student_scaled = scaler.transform(student_df)

prediction = svm_model.predict(student_scaled)
label_map = {0: "Low", 1: "Medium", 2: "High"}
print("\nPredicted stress level:", label_map[prediction[0]])

y_pred_test = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Model accuracy on test data: {accuracy * 100:.2f}%")
