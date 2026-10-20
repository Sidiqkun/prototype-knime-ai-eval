# Evaluasi KNIME AI vs Pendekatan Manual dengan Verifikasi Manusia

Kerangka eksperimen human-in-the-loop untuk menilai asisten AI pada platform low-code (KNIME) pada kasus
klasifikasi keterlambatan pembayaran invoice, dengan pemetaan kesalahan ke tahap CRISP-DM.

## Struktur
| File | Fungsi |
|---|---|
| `src/generate_data.py` | Dataset sintetik invoice (120 customer, ~4000 invoice, 2 tahun) + kolom leakage |
| `src/manual_baseline.py` | Pembanding manual: time split, 3 model, class weight, skenario leakage |
| `src/evaluate_predictions.py` | Menilai output prediksi KNIME (manual/AI) dengan metrik yang sama |
| `src/workflow_audit.py` | Audit heuristik `.knwf`: kolom leakage, split acak, node LLM |
| `src/analyze_results.py` | Ringkasan, uji Mann-Whitney, Cliff's delta, peta kesalahan per tahap CRISP-DM |
| `docs/prompts.md` | Skenario prompt dan protokol |
| `data/*_template.csv` | Template pencatatan eksperimen dan kesalahan |

## Alur
```bash
pip install -r requirements.txt
python src/generate_data.py                 # data/invoices.csv
python src/manual_baseline.py               # results/manual_metrics.csv
# --- di KNIME: bangun workflow manual & dengan KNIME AI, ekspor prediksi + .knwf ---
python src/workflow_audit.py workflows/ai_run1.knwf
python src/evaluate_predictions.py --pred out.csv --run_id AI_T1_mid_r1 --true_col late_30 --pred_col "Prediction (late_30)"
# isi data/experiments.csv dan data/errors.csv (salin dari template)
python src/analyze_results.py
```

## Catatan metodologis
- Gunakan **split waktu** yang sama untuk semua kondisi (pakai 70% awal / 30% akhir berdasarkan `invoice_date`).
- `workflow_audit.py` hanya penyaring awal; konfirmasi temuan secara manual di KNIME.
- Angka di template hanya ilustrasi, bukan hasil penelitian.
- Anonimkan data sebelum dikirim ke LLM eksternal; dokumentasikan semua prompt dan output.
