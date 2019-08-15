# Project-NN-Keras

Adi Sapto Raharjo | Magister Teknik Informatika | 1821211002

Keras adalah perpustakaan Python open source gratis yang kuat dan mudah digunakan untuk mengembangkan dan mengevaluasi model pembelajaran yang mendalam.

Ini membungkus perpustakaan perhitungan numerik yang efisien Theano dan TensorFlow dan memungkinkan Anda untuk mendefinisikan dan melatih model jaringan saraf hanya dalam beberapa baris kode.
Memuat Data

Langkah pertama adalah mendefinisikan fungsi dan kelas yang ingin kita gunakan dalam tutorial ini. Kami akan menggunakan perpustakaan NumPy untuk memuat dataset kami dan kami akan menggunakan dua kelas dari perpustakaan Keras untuk mendefinisikan model kami. Impor yang diperlukan tercantum di bawah ini.

# first neural network with keras tutorial

from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense

Dalam tutorial Keras ini, kita akan menggunakan dataset diabetes Pima Indian. Ini adalah dataset pembelajaran mesin standar dari repositori Pembelajaran Mesin UCI. Ini menggambarkan data rekam medis pasien untuk orang India Pima dan apakah mereka memiliki diabetes dalam lima tahun.

Pastikan dataset tersebut disimpan dalam komputer dan dimasukkan path atau direktori tempat dimana dataset itu disimpan.

# load the dataset

dataset = loadtxt('D:\pima-indians-diabetes.csv', delimiter=',')

​

Kita sekarang dapat memuat file sebagai matriks angka menggunakan fungsi NumPy loadtxt ().

Ada delapan variabel input dan satu variabel output (kolom terakhir). Kita akan mempelajari sebuah model untuk memetakan baris variabel input (X) ke variabel output (y), yang sering kita ringkas sebagai y = f (X).

Variabel dapat diringkas sebagai berikut: *Variabel Input (X):

    Number of times pregnant
    Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    Diastolic blood pressure (mm Hg)
    Triceps skin fold thickness (mm)
    2-Hour serum insulin (mu U/ml)
    Body mass index (weight in kg/(height in m)^2)
    Diabetes pedigree function
    Age (years)

Variabel Keluaran (y): *Variabel kelas (0 atau 1)

Setelah file CSV dimuat ke dalam memori, kita dapat membagi kolom data menjadi variabel input dan output. Data akan disimpan dalam array 2D di mana dimensi pertama adalah baris dan dimensi kedua adalah kolom, mis. [baris,kolom].

Kita dapat membagi array menjadi dua array dengan memilih subset kolom menggunakan operator slice NumPy standar atau ":" Kita dapat memilih 8 kolom pertama dari indeks 0 hingga indeks 7 melalui slice 0: 8. Kami kemudian dapat memilih kolom output (variabel ke-9) melalui indeks 8.

# split into input (X) and output (y) variables

X = dataset[:,0:8]

y = dataset[:,8]

Mendefinisikan Model Keras

Model dalam Keras didefinisikan sebagai urutan lapisan. Model Sequential dibuat dan menambahkan layer satu per satu sampai kami puas akan hasil dari arsitektur jaringan ini.

Hal pertama yang harus dilakukan adalah memastikan layer input memiliki jumlah fitur input yang tepat. Ini dapat ditentukan saat membuat layer pertama dengan argumen input_dim dan mengaturnya ke 8 untuk 8 variabel input.

Kami akan menggunakan fungsi aktivasi unit linear yang diperbaiki yang disebut sebagai ReLU pada dua lapisan pertama dan fungsi Sigmoid di lapisan output.

Dulu kasus bahwa fungsi aktivasi Sigmoid dan Tanh lebih disukai untuk semua lapisan. Saat ini, kinerja yang lebih baik dicapai menggunakan fungsi aktivasi ReLU. Kami menggunakan sigmoid pada layer output untuk memastikan output jaringan kami antara 0 dan 1 dan mudah dipetakan ke probabilitas kelas 1 atau snap ke klasifikasi keras dari kedua kelas dengan ambang batas standar 0,5.

Kita bisa menyatukan semuanya dengan menambahkan setiap layer:

-Model mengharapkan deretan data dengan 8 variabel (argumen input_dim = 8) -Lapisan tersembunyi pertama memiliki 12 node dan menggunakan fungsi aktivasi relu. -Lapisan tersembunyi kedua memiliki 8 node dan menggunakan fungsi aktivasi relu. -Lapisan output memiliki satu node dan menggunakan fungsi aktivasi sigmoid.

# define the keras model

model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

Kompilasi Keras Model

Setelah model sudah didefinisikan, kita bisa mengkompilasinya.

Mengkompilasi model menggunakan perpustakaan numerik yang efisien di bawah penutup (yang disebut backend) seperti Theano atau TensorFlow. Backend secara otomatis memilih cara terbaik untuk mewakili jaringan untuk pelatihan dan membuat prediksi untuk berjalan pada perangkat keras Anda, seperti CPU atau GPU atau bahkan didistribusikan.

Saat mengkompilasi, kita harus menentukan beberapa properti tambahan yang diperlukan saat melatih jaringan. Ingat pelatihan jaringan berarti menemukan set bobot terbaik untuk memetakan input ke output dalam dataset kami.

Kita harus menentukan fungsi kehilangan yang digunakan untuk mengevaluasi satu set bobot, optimizer digunakan untuk mencari melalui berbagai bobot untuk jaringan dan metrik opsional apa pun yang ingin kami kumpulkan dan laporkan selama pelatihan.

Dalam hal ini, kita akan menggunakan cross entropy sebagai argumen loss. Kerugian ini untuk masalah klasifikasi biner dan didefinisikan dalam Keras sebagai "binary_crossentropy".

Kami akan mendefinisikan pengoptimal sebagai algoritme gradien keturunan stokastik efisien "adam". Ini adalah versi populer dari gradient descent karena secara otomatis menyetel dirinya sendiri dan memberikan hasil yang baik dalam berbagai masalah.

karena ini adalah masalah klasifikasi, kami akan mengumpulkan dan melaporkan keakuratan klasifikasi, yang didefinisikan melalui argumen metrik.

# compile the keras model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

Fit Keras Model

Kami telah mendefinisikan model kami dan mengkompilasinya siap untuk perhitungan yang efisien. Sekarang saatnya untuk mengeksekusi model pada beberapa data.

Kita dapat melatih atau menyesuaikan model kita pada data yang dimuat dengan memanggil fungsi fit () pada model.

Pelatihan terjadi pada zaman dan setiap zaman dibagi menjadi beberapa kelompok.

    Epoch: Satu melewati semua baris dalam dataset pelatihan.
    Batch: Satu atau lebih sampel dipertimbangkan oleh model dalam epoch sebelum bobot diperbarui.

Proses pelatihan akan berjalan untuk sejumlah iterasi tetap melalui dataset yang disebut epochs, yang harus kita tentukan menggunakan argumen epochs. Kita juga harus mengatur jumlah baris dataset yang dipertimbangkan sebelum bobot model diperbarui dalam setiap zaman, yang disebut ukuran batch dan ditetapkan menggunakan argumen batch_size.

Untuk masalah ini, kita akan berlari untuk sejumlah kecil zaman (150) dan menggunakan ukuran batch yang relatif kecil yaitu 10. Ini berarti bahwa setiap zaman akan melibatkan (150/10) 15 pembaruan pada bobot model.

Konfigurasi ini dapat dipilih secara eksperimental dengan coba-coba. Kami ingin melatih model cukup sehingga ia belajar pemetaan baris input data yang baik (atau cukup baik) ke klasifikasi output. Model akan selalu memiliki beberapa kesalahan, tetapi jumlah kesalahan akan keluar setelah beberapa titik untuk konfigurasi model yang diberikan. Ini disebut model konvergensi.

# fit the keras model on the dataset

model.fit(X, y, epochs=150, batch_size=10)

WARNING:tensorflow:From D:\DOWNLOAD\Programs\Miniconda3\envs\tensorflow\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Epoch 1/150
768/768 [==============================] - 2s 3ms/step - loss: 5.3025 - acc: 0.5938
Epoch 2/150
768/768 [==============================] - 0s 163us/step - loss: 3.7186 - acc: 0.6393
Epoch 3/150
768/768 [==============================] - 0s 163us/step - loss: 3.1116 - acc: 0.6185
Epoch 4/150
768/768 [==============================] - 0s 163us/step - loss: 1.2224 - acc: 0.5911
Epoch 5/150
768/768 [==============================] - 0s 163us/step - loss: 1.0364 - acc: 0.6289
Epoch 6/150
768/768 [==============================] - 0s 163us/step - loss: 0.9374 - acc: 0.6263
Epoch 7/150
768/768 [==============================] - 0s 183us/step - loss: 0.8878 - acc: 0.6237
Epoch 8/150
768/768 [==============================] - 0s 163us/step - loss: 0.7942 - acc: 0.6432
Epoch 9/150
768/768 [==============================] - 0s 142us/step - loss: 0.7665 - acc: 0.6263
Epoch 10/150
768/768 [==============================] - 0s 183us/step - loss: 0.7429 - acc: 0.6393
Epoch 11/150
768/768 [==============================] - 0s 183us/step - loss: 0.7069 - acc: 0.6497
Epoch 12/150
768/768 [==============================] - 0s 183us/step - loss: 0.6857 - acc: 0.6615
Epoch 13/150
768/768 [==============================] - 0s 183us/step - loss: 0.6792 - acc: 0.6484
Epoch 14/150
768/768 [==============================] - 0s 163us/step - loss: 0.6504 - acc: 0.6484
Epoch 15/150
768/768 [==============================] - 0s 142us/step - loss: 0.6513 - acc: 0.6536
Epoch 16/150
768/768 [==============================] - 0s 142us/step - loss: 0.6395 - acc: 0.6797
Epoch 17/150
768/768 [==============================] - 0s 163us/step - loss: 0.6322 - acc: 0.6797
Epoch 18/150
768/768 [==============================] - 0s 142us/step - loss: 0.6174 - acc: 0.6888
Epoch 19/150
768/768 [==============================] - 0s 142us/step - loss: 0.6260 - acc: 0.6784
Epoch 20/150
768/768 [==============================] - 0s 163us/step - loss: 0.6003 - acc: 0.6888
Epoch 21/150
768/768 [==============================] - 0s 183us/step - loss: 0.6005 - acc: 0.6940
Epoch 22/150
768/768 [==============================] - 0s 203us/step - loss: 0.6091 - acc: 0.6771
Epoch 23/150
768/768 [==============================] - 0s 203us/step - loss: 0.5813 - acc: 0.7031
Epoch 24/150
768/768 [==============================] - 0s 325us/step - loss: 0.5904 - acc: 0.7070
Epoch 25/150
768/768 [==============================] - 0s 366us/step - loss: 0.5771 - acc: 0.7122
Epoch 26/150
768/768 [==============================] - 0s 264us/step - loss: 0.5979 - acc: 0.6823
Epoch 27/150
768/768 [==============================] - 0s 142us/step - loss: 0.5733 - acc: 0.7122
Epoch 28/150
768/768 [==============================] - 0s 142us/step - loss: 0.5818 - acc: 0.6979
Epoch 29/150
768/768 [==============================] - 0s 163us/step - loss: 0.5841 - acc: 0.6901
Epoch 30/150
768/768 [==============================] - 0s 244us/step - loss: 0.5872 - acc: 0.6888
Epoch 31/150
768/768 [==============================] - 0s 183us/step - loss: 0.5628 - acc: 0.7018
Epoch 32/150
768/768 [==============================] - 0s 142us/step - loss: 0.5665 - acc: 0.7240
Epoch 33/150
768/768 [==============================] - 0s 142us/step - loss: 0.5569 - acc: 0.7292
Epoch 34/150
768/768 [==============================] - 0s 163us/step - loss: 0.5764 - acc: 0.6979
Epoch 35/150
768/768 [==============================] - 0s 142us/step - loss: 0.5651 - acc: 0.7174
Epoch 36/150
768/768 [==============================] - 0s 122us/step - loss: 0.5583 - acc: 0.7148
Epoch 37/150
768/768 [==============================] - 0s 183us/step - loss: 0.5732 - acc: 0.6966
Epoch 38/150
768/768 [==============================] - 0s 183us/step - loss: 0.5512 - acc: 0.7083
Epoch 39/150
768/768 [==============================] - 0s 142us/step - loss: 0.5593 - acc: 0.7135
Epoch 40/150
768/768 [==============================] - 0s 183us/step - loss: 0.5724 - acc: 0.7005
Epoch 41/150
768/768 [==============================] - 0s 224us/step - loss: 0.5564 - acc: 0.7201
Epoch 42/150
768/768 [==============================] - 0s 203us/step - loss: 0.5580 - acc: 0.7292
Epoch 43/150
768/768 [==============================] - 0s 224us/step - loss: 0.5709 - acc: 0.7135
Epoch 44/150
768/768 [==============================] - 0s 183us/step - loss: 0.5604 - acc: 0.7201
Epoch 45/150
768/768 [==============================] - 0s 183us/step - loss: 0.5558 - acc: 0.7201
Epoch 46/150
768/768 [==============================] - 0s 203us/step - loss: 0.5488 - acc: 0.7161
Epoch 47/150
768/768 [==============================] - 0s 163us/step - loss: 0.5489 - acc: 0.7214
Epoch 48/150
768/768 [==============================] - 0s 163us/step - loss: 0.5477 - acc: 0.7135
Epoch 49/150
768/768 [==============================] - 0s 163us/step - loss: 0.5542 - acc: 0.7266
Epoch 50/150
768/768 [==============================] - 0s 183us/step - loss: 0.5523 - acc: 0.7188
Epoch 51/150
768/768 [==============================] - 0s 224us/step - loss: 0.5579 - acc: 0.7018
Epoch 52/150
768/768 [==============================] - 0s 203us/step - loss: 0.5840 - acc: 0.6901
Epoch 53/150
768/768 [==============================] - 0s 183us/step - loss: 0.5546 - acc: 0.7109
Epoch 54/150
768/768 [==============================] - 0s 142us/step - loss: 0.5574 - acc: 0.7044
Epoch 55/150
768/768 [==============================] - 0s 142us/step - loss: 0.5569 - acc: 0.7109
Epoch 56/150
768/768 [==============================] - 0s 183us/step - loss: 0.5443 - acc: 0.7122
Epoch 57/150
768/768 [==============================] - 0s 183us/step - loss: 0.5418 - acc: 0.7344
Epoch 58/150
768/768 [==============================] - 0s 183us/step - loss: 0.5449 - acc: 0.7266
Epoch 59/150
768/768 [==============================] - 0s 122us/step - loss: 0.5431 - acc: 0.7188
Epoch 60/150
768/768 [==============================] - 0s 142us/step - loss: 0.5347 - acc: 0.7409
Epoch 61/150
768/768 [==============================] - 0s 122us/step - loss: 0.5503 - acc: 0.7240
Epoch 62/150
768/768 [==============================] - 0s 122us/step - loss: 0.5509 - acc: 0.7266
Epoch 63/150
768/768 [==============================] - 0s 122us/step - loss: 0.5497 - acc: 0.7292
Epoch 64/150
768/768 [==============================] - 0s 122us/step - loss: 0.5289 - acc: 0.7396
Epoch 65/150
768/768 [==============================] - 0s 142us/step - loss: 0.5664 - acc: 0.7174
Epoch 66/150
768/768 [==============================] - 0s 122us/step - loss: 0.5812 - acc: 0.7057
Epoch 67/150
768/768 [==============================] - 0s 163us/step - loss: 0.5318 - acc: 0.7331
Epoch 68/150
768/768 [==============================] - 0s 183us/step - loss: 0.5223 - acc: 0.7552
Epoch 69/150
768/768 [==============================] - 0s 142us/step - loss: 0.5427 - acc: 0.7174
Epoch 70/150
768/768 [==============================] - 0s 142us/step - loss: 0.5369 - acc: 0.7148
Epoch 71/150
768/768 [==============================] - 0s 142us/step - loss: 0.5289 - acc: 0.7253
Epoch 72/150
768/768 [==============================] - 0s 142us/step - loss: 0.5428 - acc: 0.7070
Epoch 73/150
768/768 [==============================] - 0s 122us/step - loss: 0.5297 - acc: 0.7383
Epoch 74/150
768/768 [==============================] - 0s 142us/step - loss: 0.5225 - acc: 0.7292
Epoch 75/150
768/768 [==============================] - 0s 142us/step - loss: 0.5296 - acc: 0.7357
Epoch 76/150
768/768 [==============================] - 0s 163us/step - loss: 0.5331 - acc: 0.7383
Epoch 77/150
768/768 [==============================] - 0s 163us/step - loss: 0.5345 - acc: 0.7305
Epoch 78/150
768/768 [==============================] - 0s 142us/step - loss: 0.5186 - acc: 0.7370
Epoch 79/150
768/768 [==============================] - 0s 183us/step - loss: 0.5253 - acc: 0.7448
Epoch 80/150
768/768 [==============================] - 0s 163us/step - loss: 0.5443 - acc: 0.7148
Epoch 81/150

768/768 [==============================] - 0s 142us/step - loss: 0.5203 - acc: 0.7461
Epoch 82/150
768/768 [==============================] - 0s 163us/step - loss: 0.5200 - acc: 0.7318
Epoch 83/150
768/768 [==============================] - 0s 122us/step - loss: 0.5271 - acc: 0.7370
Epoch 84/150
768/768 [==============================] - 0s 163us/step - loss: 0.5189 - acc: 0.7461
Epoch 85/150
768/768 [==============================] - 0s 122us/step - loss: 0.5190 - acc: 0.7279
Epoch 86/150
768/768 [==============================] - 0s 142us/step - loss: 0.5362 - acc: 0.7305
Epoch 87/150
768/768 [==============================] - 0s 122us/step - loss: 0.5397 - acc: 0.7292
Epoch 88/150
768/768 [==============================] - 0s 203us/step - loss: 0.5262 - acc: 0.7357
Epoch 89/150
768/768 [==============================] - 0s 183us/step - loss: 0.5262 - acc: 0.7435
Epoch 90/150
768/768 [==============================] - 0s 163us/step - loss: 0.5247 - acc: 0.7396
Epoch 91/150
768/768 [==============================] - 0s 163us/step - loss: 0.5186 - acc: 0.7331
Epoch 92/150
768/768 [==============================] - 0s 142us/step - loss: 0.5223 - acc: 0.7357
Epoch 93/150
768/768 [==============================] - 0s 142us/step - loss: 0.5143 - acc: 0.7409
Epoch 94/150
768/768 [==============================] - 0s 142us/step - loss: 0.5265 - acc: 0.7448
Epoch 95/150
768/768 [==============================] - 0s 203us/step - loss: 0.5373 - acc: 0.7409
Epoch 96/150
768/768 [==============================] - 0s 142us/step - loss: 0.5158 - acc: 0.7448
Epoch 97/150
768/768 [==============================] - 0s 122us/step - loss: 0.5152 - acc: 0.7474
Epoch 98/150
768/768 [==============================] - 0s 142us/step - loss: 0.5241 - acc: 0.7396
Epoch 99/150
768/768 [==============================] - 0s 142us/step - loss: 0.5152 - acc: 0.7539
Epoch 100/150
768/768 [==============================] - 0s 122us/step - loss: 0.5184 - acc: 0.7383
Epoch 101/150
768/768 [==============================] - 0s 142us/step - loss: 0.5127 - acc: 0.7474
Epoch 102/150
768/768 [==============================] - 0s 122us/step - loss: 0.5152 - acc: 0.7344
Epoch 103/150
768/768 [==============================] - 0s 122us/step - loss: 0.5087 - acc: 0.7409
Epoch 104/150
768/768 [==============================] - 0s 122us/step - loss: 0.5131 - acc: 0.7565
Epoch 105/150
768/768 [==============================] - 0s 122us/step - loss: 0.5143 - acc: 0.7331
Epoch 106/150
768/768 [==============================] - 0s 122us/step - loss: 0.5096 - acc: 0.7500
Epoch 107/150
768/768 [==============================] - 0s 142us/step - loss: 0.5017 - acc: 0.7526
Epoch 108/150
768/768 [==============================] - 0s 122us/step - loss: 0.5087 - acc: 0.7513
Epoch 109/150
768/768 [==============================] - 0s 142us/step - loss: 0.5044 - acc: 0.7526
Epoch 110/150
768/768 [==============================] - 0s 122us/step - loss: 0.5209 - acc: 0.7383
Epoch 111/150
768/768 [==============================] - 0s 122us/step - loss: 0.4915 - acc: 0.7526
Epoch 112/150
768/768 [==============================] - 0s 122us/step - loss: 0.5239 - acc: 0.7435
Epoch 113/150
768/768 [==============================] - 0s 122us/step - loss: 0.5192 - acc: 0.7409
Epoch 114/150
768/768 [==============================] - 0s 142us/step - loss: 0.4975 - acc: 0.7526
Epoch 115/150
768/768 [==============================] - 0s 122us/step - loss: 0.5044 - acc: 0.7487
Epoch 116/150
768/768 [==============================] - 0s 122us/step - loss: 0.5137 - acc: 0.7409
Epoch 117/150
768/768 [==============================] - 0s 142us/step - loss: 0.4952 - acc: 0.7591
Epoch 118/150
768/768 [==============================] - 0s 122us/step - loss: 0.4952 - acc: 0.7513
Epoch 119/150
768/768 [==============================] - 0s 142us/step - loss: 0.5114 - acc: 0.7513
Epoch 120/150
768/768 [==============================] - 0s 122us/step - loss: 0.4949 - acc: 0.7435
Epoch 121/150
768/768 [==============================] - 0s 122us/step - loss: 0.4930 - acc: 0.7487
Epoch 122/150
768/768 [==============================] - 0s 122us/step - loss: 0.5060 - acc: 0.7422
Epoch 123/150
768/768 [==============================] - 0s 142us/step - loss: 0.4951 - acc: 0.7669
Epoch 124/150
768/768 [==============================] - 0s 122us/step - loss: 0.4973 - acc: 0.7383
Epoch 125/150
768/768 [==============================] - 0s 122us/step - loss: 0.5075 - acc: 0.7578
Epoch 126/150
768/768 [==============================] - 0s 122us/step - loss: 0.5147 - acc: 0.7318
Epoch 127/150
768/768 [==============================] - 0s 203us/step - loss: 0.4988 - acc: 0.7578
Epoch 128/150
768/768 [==============================] - 0s 183us/step - loss: 0.4864 - acc: 0.7565
Epoch 129/150
768/768 [==============================] - 0s 203us/step - loss: 0.4911 - acc: 0.7656
Epoch 130/150
768/768 [==============================] - 0s 163us/step - loss: 0.4987 - acc: 0.7656
Epoch 131/150
768/768 [==============================] - 0s 142us/step - loss: 0.4945 - acc: 0.7565
Epoch 132/150
768/768 [==============================] - 0s 142us/step - loss: 0.4959 - acc: 0.7539
Epoch 133/150
768/768 [==============================] - 0s 183us/step - loss: 0.4955 - acc: 0.7500
Epoch 134/150
768/768 [==============================] - 0s 183us/step - loss: 0.4954 - acc: 0.7565
Epoch 135/150
768/768 [==============================] - 0s 163us/step - loss: 0.4848 - acc: 0.7630
Epoch 136/150
768/768 [==============================] - 0s 142us/step - loss: 0.4870 - acc: 0.7708
Epoch 137/150
768/768 [==============================] - 0s 122us/step - loss: 0.5180 - acc: 0.7539
Epoch 138/150
768/768 [==============================] - 0s 142us/step - loss: 0.5073 - acc: 0.7526
Epoch 139/150
768/768 [==============================] - 0s 142us/step - loss: 0.5114 - acc: 0.7656
Epoch 140/150
768/768 [==============================] - 0s 142us/step - loss: 0.4901 - acc: 0.7591
Epoch 141/150
768/768 [==============================] - 0s 142us/step - loss: 0.5028 - acc: 0.7643
Epoch 142/150
768/768 [==============================] - 0s 142us/step - loss: 0.5159 - acc: 0.7526
Epoch 143/150
768/768 [==============================] - 0s 142us/step - loss: 0.4928 - acc: 0.7474
Epoch 144/150
768/768 [==============================] - 0s 122us/step - loss: 0.4934 - acc: 0.7539
Epoch 145/150
768/768 [==============================] - 0s 142us/step - loss: 0.4976 - acc: 0.7656
Epoch 146/150
768/768 [==============================] - 0s 122us/step - loss: 0.4979 - acc: 0.7617
Epoch 147/150
768/768 [==============================] - 0s 122us/step - loss: 0.4845 - acc: 0.7682
Epoch 148/150
768/768 [==============================] - 0s 142us/step - loss: 0.4995 - acc: 0.7578
Epoch 149/150
768/768 [==============================] - 0s 183us/step - loss: 0.5019 - acc: 0.7539
Epoch 150/150
768/768 [==============================] - 0s 183us/step - loss: 0.4874 - acc: 0.7591

<keras.callbacks.History at 0x295acfe0be0>

Evaluasi Model Keras

Kami telah melatih jaringan saraf kami pada seluruh dataset dan kami dapat mengevaluasi kinerja jaringan pada dataset yang sama.

Ini hanya akan memberi kita gambaran tentang seberapa baik kita telah memodelkan dataset (mis. Akurasi kereta), tetapi tidak tahu seberapa baik algoritma dapat bekerja pada data baru. Kami telah melakukan ini untuk kesederhanaan, tetapi idealnya, Anda dapat memisahkan data Anda ke dalam set data kereta dan uji untuk pelatihan dan evaluasi model Anda.

Anda dapat mengevaluasi model Anda pada dataset pelatihan Anda menggunakan fungsi evalu () pada model Anda dan memberikan input dan output yang sama dengan yang digunakan untuk melatih model.

Ini akan menghasilkan prediksi untuk setiap pasangan input dan output dan mengumpulkan skor, termasuk kehilangan rata-rata dan metrik apa pun yang telah Anda konfigurasi, seperti akurasi.

Fungsi evaluate () akan mengembalikan daftar dengan dua nilai. Yang pertama adalah hilangnya model pada dataset dan yang kedua adalah akurasi model pada dataset. Kami hanya tertarik untuk melaporkan keakuratan, sehingga kami akan mengabaikan nilai kerugiannya.

Berikut adalah hasil akurasi yang diambil dari fungsi evaluate().

# evaluate the keras model

_, accuracy = model.evaluate(X, y)

print('Accuracy: %.2f' % (accuracy*100))

768/768 [==============================] - 0s 203us/step
Accuracy: 77.99

Akurasi tersebut akan berubah apabila dataset diatas diulangi proses fit-nya.
Selesai

Terima kasih, mohon maaf atas kekurangan atau kesalahan kata.

Disusun dan Ditulis oleh Adi Sapto Raharjo
