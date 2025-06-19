if 'Loan_Status' in test_clean.columns:
    test_X = test_clean.drop('Loan_Status', axis=1)
else:
    test_X = test_clean.copy()

# Prediksi
test_pred = model.predict(test_X)

# Konversi ke Y/N
prediksi = pd.DataFrame({
    'Loan_Status_Predicted': ['Y' if val == 1 else 'N' for val in test_pred]
})

# Simpan ke file
prediksi.to_csv("hasil_prediksi_test.csv", index=False)

from google.colab import files
files.download("hasil_prediksi_test.csv")

"""Analisis Kelayakan Kredit Nasabah dengan Pohon Keputusan"""

from sklearn.tree import export_text
print(export_text(model, feature_names=list(X.columns)))