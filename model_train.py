from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Pisahkan fitur dan target
X = train_clean.drop('Loan_Status', axis=1)
y = train_clean['Loan_Status']

# Jika target masih huruf (Y/N), ubah ke 1/0
if y.dtype == 'object':
    y = y.map({'Y': 1, 'N': 0})

# Split dataset untuk validasi
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan training Decision Tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_val)

# Evaluasi
print("✅ Evaluasi Model:")
print("Akurasi:", accuracy_score(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))