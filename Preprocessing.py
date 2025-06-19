from google.colab import files
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import io

print("⬆️ Upload file train.csv")
uploaded_train = files.upload()

print("⬆️ Upload file test.csv")
uploaded_test = files.upload()

train_df = pd.read_csv(io.BytesIO(uploaded_train[list(uploaded_train.keys())[0]]))
test_df = pd.read_csv(io.BytesIO(uploaded_test[list(uploaded_test.keys())[0]]))

print("✅ File berhasil dimuat!")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

def preprocess(df):
    df = df.copy()

    # Isi missing value untuk kolom numerik & kategorikal
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # Encode kategori menjadi angka
    label_cols = ['Gender', 'Married', 'Education', 'Self_Employed',
                  'Property_Area', 'Dependents']

    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Jika kolom 'Loan_ID' atau target ada, pisahkan
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)

    return df

# Terapkan preprocessing ke train dan test
train_clean = preprocess(train_df)
test_clean = preprocess(test_df)

print("✅ Preprocessing selesai!")
train_clean.head()

train_clean.to_csv("train_cleaned.csv", index=False)
test_clean.to_csv("test_cleaned.csv", index=False)

files.download("train_cleaned.csv")
files.download("test_cleaned.csv")

