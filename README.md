# Laporan Proyek Machine Learning - Annisa Mufidatun Sholihah

## Domain Proyek

![Diabetes effect](https://assets.mrmed.in/others/file-1632987112542-597612264-Side%20effects%20of%20Diabetes.jpeg?w=1920&q=75)

Diabetes melitus, umumnya dikenal sebagai diabetes, merupakan penyakit kronis yang serius. Kondisi ini muncul ketika kadar glukosa darah meningkat akibat ketidakmampuan tubuh menghasilkan insulin secara cukup, atau ketidakefektifan dalam menggunakan insulin yang diproduksi. Diabetes melitus menjadi ancaman global yang signifikan terhadap kesehatan, tanpa memandang status sosial ekonomi atau batas negara. Saat ini terdapat 463 juta orang dewasa yang hidup dengan diabetes melitus. Jika tidak ada langkah-langkah yang tepat untuk mengatasi ini, diperkirakan jumlah penderita akan mencapai 578 juta pada tahun 2030. Lebih mengkhawatirkan lagi, angka tersebut diprediksi akan melonjak hingga 700 juta pada tahun 2045 [2].

Diabetes Melitus adalah kondisi yang mengerikan menurut laporan WHO tahun 2016, langkah-langkah penting diperlukan untuk mencegah dan mengobati penyakit ini. Diabetes Tipe I adalah jenis Diabetes paling umum yang terjadi pada kelompok usia yang lebih muda. Penyakit ini meningkat pesat di seluruh dunia karena perubahan gaya hidup dan pola makan yang tidak sehat. Menurut perkiraan Federasi Diabetes Internasional (IDF), prevalensi Diabetes melitus kemungkinan akan meningkat setiap tahunnya. Kasus Diabetes Tipe 2 menjadi lebih menonjol pada usia yang lebih muda dan negara-negara berkembang, mencakup 85-95% dari pasien Diabetes melitus. Penyakit ini terjadi pada kelompok usia antara 18-75 tahun; sekitar 285 juta orang menderita Diabetes di seluruh dunia, menurut survei yang dilakukan pada tahun 2010. Pada tahun 2025, diperkirakan 438 juta orang di negara-negara berkembang akan meningkat dengan tingkat 60-75% dan meningkatkan angka kematian hingga sekitar 60%, yang bisa berakibat fatal [1].


## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Diabetes Melitus adalah penyakit kronis yang menyebabkan peningkatan kadar glukosa darah akibat ketidakmampuan tubuh untuk memproduksi atau menggunakan insulin secara efektif. Penyakit ini menjadi ancaman global yang terus meningkat tanpa memandang status sosial ekonomi atau batas negara.
- Prevalensi diabetes melitus diperkirakan akan meningkat secara signifikan dalam beberapa tahun ke depan, dengan jumlah penderita yang diperkirakan mencapai 578 juta pada tahun 2030 dan 700 juta pada tahun 2045. Peningkatan ini menimbulkan kebutuhan mendesak untuk alat prediksi yang efektif.
- Kasus diabetes tipe 2 semakin umum di kalangan usia muda dan di negara-negara berkembang, yang mengindikasikan perlunya deteksi dini dan pencegahan. Model prediksi yang akurat dapat membantu dalam mengidentifikasi risiko diabetes lebih awal, sebelum gejala menjadi lebih parah.

### Goals

- Membangun model machine learning yang dapat memprediksi risiko diabetes melitus dengan akurat berdasarkan data medis dan gejala, sehingga memberikan alat yang berguna untuk deteksi dini dan intervensi.
- Mengembangkan model yang dapat memproyeksikan risiko diabetes dengan tingkat akurasi yang tinggi untuk membantu mengidentifikasi pasien berisiko sebelum prevalensi meningkat secara drastis, serta mendukung langkah-langkah preventif yang tepat.
- Melakukan analisis terhadap fitur-fitur yang berkontribusi pada risiko diabetes dan menerapkan model yang dapat diintegrasikan dalam sistem kesehatan untuk mendukung diagnosis dan perawatan, serta meningkatkan kesadaran dan langkah-langkah pencegahan di populasi berisiko.


### Solution statements
Untuk mencapai tujuan tersebut, kita mengajukan beberapa solusi yang dapat diukur dengan metrik evaluasi akurasi, presisi, recall, dan F1-score.

*   Solution 1: Logistic Regression
*   Solution 2: Random Forest

Dengan menggunakan kedua solusi ini, kita dapat membandingkan kinerja model logistic regression dan random forest untuk menentukan pendekatan mana yang lebih efektif dalam memprediksi risiko diabetes pada tahap awal.

## Data Understanding

**Dataset Early Stage Diabetes Risk Prediction Dataset**
Dataset ini berisi informasi mengenai tanda dan gejala pasien diabetes yang baru terdiagnosis atau mereka yang berisiko terkena diabetes. Data dikumpulkan melalui kuesioner langsung yang diberikan kepada pasien di Rumah Sakit Diabetes Sylhet di Sylhet, Bangladesh, dan disetujui oleh dokter. Contoh: [Link to dataset](https://www.kaggle.com/datasets/abdelazizsami/early-stage-diabetes-risk-prediction).

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

- **Age** (integer): Umur dari pasien.
- **Gender** (categorical): Jenis kelamin pasien.
- **Polyuria** (binary): Apakah pasien buang air kecil lebih sering dibanding hari biasanya.
- **Polydipsia** (binary): Apakah pasien merasa haus yang tidak berkesudahan.
- **Sudden Weight Loss** (binary): Apakah pasien mengalami penurunan berat badan yang signifikan.
- **Weakness** (binary): Apakah pasien mengalami kelemahan.
- **Polyphagia** (binary): Apakah pasien merasa lapar ekstrem yang tidak terpuaskan meskipun sudah makan.
- **Genital Thrush** (binary): Apakah pasien mengalami infeksi jamur pada genital/alat kelamin.
- **Visual Blurring** (binary): Apakah pasien penglihatannya kabur.
- **Itching** (binary): Apakah pasien mengalami gatal.
- **Irritability** (binary): Apakah pasien merasakan perasaan gelisah yang mungkin Anda alami akibat stres, kondisi kesehatan mental, atau gangguan fisik.
- **Delayed Healing** (binary): Apakah pasien mengalami penyembuhan luka yang lebih lambat dari biasanya.
- **Partial Paresis** (binary): Apakah pasien mengalami kelemahan atau kelumpuhan sebagian pada area tubuh tertentu.
- **Muscle Stiffness** (binary): Apakah pasien merasakan kekakuan atau ketegangan pada otot-otot tubuh.
- **Alopecia** (binary): Apakah pasien mengalami kerontokan rambut yang tidak normal atau kebotakan.
- **Obesity** (binary): Apakah pasien memiliki berat badan berlebih atau obesitas.
- **Class** (binary): Parameter apakah pasien mengalami diabetes atau tidak.


**Exploratory Data Analysis**
Untuk memahami data, dilakukan beberapa cara yaitu 
- melihat tipe data setiap fitur dengan **data.info()**
- melihat informasi statistik dari fitur numerik yaitu age dengan **data.describe()**
- melihat apakah ada nilai null dalam dataset dengan **data.isnull().sum()**
- Melakukan univariate analysis dengan melakukan visualisasi data pada setiap fitur. Untuk fitur numerik digunakan histogram dan data kategorikal dengan bar chart.


## Data Preparation
Pada dataset dilakukan beberapa proses berikut
1.  Encoding
    Encoding dilakukan karena model machine learning membutuhkan input data dalam bentuk numerik untuk melakukan perhitungan dan prediksi.Encoding dilakukan pada categorical feature yaitu Gender, Polyuria,Polydipsia, sudden weight loss, weakness, Polyphagia, Genital thrush, visual blurring, Itching, Irritability, delayed healing, partial paresis, muscle stiffness, Alopecia, Obesity, dan class
    Encoding dilakukan sebagai berikut
    Yes = 1; No = 0
    Male = 0; Female = 1
    Negative = 0; Positive = 1

2.  Menghapus outlier
    Pada feature age outlier dihapus. Outlier dapat mempengaruhi kinerja model machine learning dengan membuat model menjadi terlalu kompleks atau terlalu sederhana. Menghapus outlier dapat membantu model untuk generalisasi lebih baik pada data yang sebenarnya, meningkatkan akurasi prediksi.
    Ada 4 baris data yang dihapus yaitu data dengan umur 85 dan 90.
    
3.  Standar Scaler
    Pada fitur umur dilakukan standar scaler, mengubah fitur sehingga memiliki rata-rata 0 dan standar deviasi 1, untuk meningkatkan kinerja dan konvergensi model.

4.  Membagi dataset menjadi 80% data training dan 20% data testing


## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

Metode yang digunakan untuk membangun model adalah **logistic regression** dan **random forest**.
Logistic Regression
Kelebihan:
- Efisien Secara Komputasi:
Logistic Regression relatif cepat untuk dilatih dan dieksekusi, bahkan pada dataset yang besar.
- Baik untuk Data Linear
Logistic Regression bekerja sangat baik ketika hubungan antara fitur dan label adalah linear atau mendekati linear.

Kekurangan:
- Asumsi Independen Fitur:
Logistic Regression mengasumsikan bahwa fitur-fitur input independen, yang tidak selalu sesuai dengan kenyataan.
- Sensitif terhadap Outlier
Model ini dapat dipengaruhi oleh outlier, yang dapat mendistorsi hasil prediksi.

Random Forest
Kelebihan:
- Mengurangi Overfitting
Dengan menggabungkan prediksi dari banyak pohon keputusan (trees), Random Forest mengurangi risiko overfitting yang sering terjadi pada pohon keputusan tunggal.
- Robust terhadap Outlier
Karena menggunakan banyak pohon, Random Forest lebih robust terhadap outlier dibandingkan dengan model lain.
- Feature Importance
Random Forest memberikan informasi tentang pentingnya fitur-fitur dalam prediksi, yang berguna untuk interpretasi dan pemahaman model.

Kekurangan:

- Kompleks dan Sulit Diinterpretasi
Random Forest adalah model yang kompleks dan sulit untuk diinterpretasi dibandingkan dengan model yang lebih sederhana seperti Logistic Regression.
- Waktu dan Sumber Daya Komputasi
Random Forest membutuhkan lebih banyak waktu dan sumber daya komputasi untuk dilatih dan dieksekusi, terutama pada dataset yang besar.
- Memori yang Dibutuhkan:
Karena menyimpan banyak pohon keputusan, Random Forest bisa sangat intensif dalam penggunaan memori.

## Evaluation
Untuk menganalisis hasil model digunakna metrik **akurasi, precision, recall, dan F1 score**. 

#### Akurasi (Accuracy)
- **Definisi**: Akurasi mengukur proporsi prediksi yang benar terhadap keseluruhan prediksi.
- **Formula**: 
  \[
  \text{Akurasi} = \frac{\text{True Positives (TP)} + \text{True Negatives (TN)}}{\text{Total Predictions}}
  \]

#### Precision (Presisi)

- **Definisi**: Presisi mengukur proporsi prediksi positif yang benar terhadap semua prediksi positif.
- **Formula**: 
  \[
  \text{Presisi} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
  \]


#### Recall (Recall) atau Sensitivitas

- **Definisi**: Recall mengukur proporsi prediksi positif yang benar terhadap semua sampel yang sebenarnya positif.
- **Formula**: 
  \[
  \text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
  \]


#### F1 Score

- **Definisi**: F1 Score adalah rata-rata harmonis dari presisi dan recall, yang memberikan keseimbangan antara keduanya.
- **Formula**: 
  \[
  \text{F1 Score} = 2 \cdot \frac{\text{Presisi} \cdot \text{Recall}}{\text{Presisi} + \text{Recall}}
  \]


Kedua model dievaluasi dengan metrik evaluasi akurasi, presisi, recall, dan F1-score dan didapatkan hasil sebagai berikut
![Hasil](https://drive.google.com/file/d/1BvO-pxbc2498_QaK9DVXht3f_7BFL_5Z)

Dapat dilihat bahwa model dengan menggunakan metode random forest memiliki hasil yang lebih baik

## Referensi

[1] [Ahuja, Ashima & Gupta, Reena & Gupta, Jitendra. (2020). Diabetes Silent Killer: Medical focus on Food Replacement and Dietary Plans. Advances in Bioresearch. 11. 128-135. 10.15515/abr.0976-4585.11.5.128135. ](https://www.researchgate.net/publication/344901853_Diabetes_Silent_Killer_Medical_focus_on_Food_Replacement_and_Dietary_Plans)
[2] [IDF. 2019. IDF Diabetes Atlas 9th Edition.](https://diabetesatlas.org/atlas/ninth-edition/)


**---Ini adalah bagian akhir laporan---**
