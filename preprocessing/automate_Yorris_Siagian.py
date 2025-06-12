import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

def preprocess_students_performance(
    input_path=Path(r'C:\Eksperimen_SML_Yorris_Siagian\StudentsPerformance.csv'),
    output_path=Path(r'C:\Eksperimen_SML_Yorris_Siagian\students_performance_preprocessing.csv')):
    
    # Load dataset
    df = pd.read_csv(input_path)

    # Encode categorical columns
    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Create new feature: average_score
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

    # Scale numerical features
    scaler = StandardScaler()
    scaled_cols = ['math score', 'reading score', 'writing score', 'average_score']
    df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

    # Save the preprocessed dataset
    df.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    preprocess_students_performance()
