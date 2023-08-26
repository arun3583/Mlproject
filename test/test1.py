from src.utils import load_object
import pandas as pd
preprocessor=load_object(file_path='artifact/preprocessor.pkl')
data = {
    'writing_score': [56.0],
    'reading_score': [76.0],
    'gender': ['female'],
    'race_ethnicity': [None],
    'parental_level_of_education': ["bachelor's degree"],
    'lunch': ['free/reduced'],
    'test_preparation_course': ['completed']
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)
data_scaled=preprocessor.transform(df)
model=load_object(file_path='artifact/model.pkl')
preds=model.predict(data_scaled)
print(preds[0])