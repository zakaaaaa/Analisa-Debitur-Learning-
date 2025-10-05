Loan Approval Prediction:Machine Learning Project menggunakan Decision Tree Classifier
Deskripsi:
Project ini merupakan model machine learning berbasis algoritma Decision Tree Classifier yang digunakan untuk memprediksi kelayakan persetujuan pinjaman (Loan Approval). Model dilatih menggunakan dataset terstruktur yang berisi berbagai fitur seperti pendapatan, status perkawinan, jumlah pinjaman, dan riwayat kredit.
Tujuan dari sistem ini adalah untuk melakukan klasifikasi biner — menentukan apakah pengajuan pinjaman disetujui (1) atau ditolak (0) berdasarkan parameter input.
Teknologi yang Digunakan:
1. Python 3.x
2. scikit-learn – untuk model Decision Tree dan evaluasi performa
3. pandas – untuk manajemen dan pembersihan dataset
4. NumPy – untuk manipulasi numerik dan array
Metodologi:
1.Preprocessing Data
Dataset dibersihkan dan dipisahkan menjadi fitur (X) dan target (y). Label Loan_Status dikonversi dari 'Y'/'N' menjadi 1/0 untuk kompatibilitas model.
2.Training dan Validasi
Data dibagi menggunakan train_test_split (80% train, 20% validation).
3.Model Training
Model DecisionTreeClassifier dengan kriteria entropy dan max_depth=5 digunakan untuk mencegah overfitting.
4.Evaluasi Model
Performa dievaluasi menggunakan Accuracy Score, Confusion Matrix, dan Classification Report.
