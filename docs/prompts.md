# Skenario Prompt (diulang minimal 5x per level, catat versi KNIME & model)

**short**: "Buat workflow prediksi invoice terlambat dari tabel ini."

**medium**: "Target: late_30 (1 = belum terbayar >30 hari setelah jatuh tempo). Buat workflow klasifikasi
dan evaluasi hasilnya."

**detailed**: "Target late_30. Buang kolom yang tidak tersedia saat invoice terbit (paid_date, days_overdue,
final_status). Pisahkan data berdasarkan invoice_date (70% awal latih, 30% akhir uji), bandingkan regresi
logistik, decision tree, random forest, laporkan recall, F1, ROC-AUC dan confusion matrix."

## Protokol
1. Anonimkan data sebelum dikirim ke layanan AI.
2. Simpan screenshot chat + .knwf tiap run.
3. Jalankan `workflow_audit.py`, lalu evaluasi dengan `evaluate_predictions.py`.
4. Catat waktu (menit), kesalahan (errors.csv), dan koreksi manual (experiments.csv).
5. Penilai kedua mengisi errors.csv secara independen, lalu hitung Cohen's kappa.
