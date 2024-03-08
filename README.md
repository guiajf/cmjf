# Modelos de rede neural para diagnóstico de malária

## Introdução


Modelos de rede neural convolucional foram implementados para detectar o *Plasmodium*, parasita responsável pela transmissão da malária, em um conjunto de imagens de células sanguíneas, especificamente das hemácias.

A malária é uma infecção dos glóbulos vermelhos do sangue causada por uma das cinco espécies de protozoários *Plasmodium*.

O método diagnóstico mais comum é o exame microscópico de amostras de sangue, que depende da habilidade técnica do profissional.

As redes neurais convolucionais densamente conectadas podem ser utilizadas para extrair características e classificar imagens, visando detectar o parasita responsável pela transmissão da malária.

## Dataset

O dataset consiste em 27.558 imagens de células, equilibrado com igual quantidade de células positivas e negativas, ou seja, infectadas e não infectadas. Ele está disponível no *National Library of Medicine* dos Estados Unidos e pode ser baixado em: https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip

## Indicadores de base

O código em Python foi executado no Google Colab para reproduzir modelos de rede neural convolucional (CNN), utilizando a API Keras do TensorFlow, capazes de classificar imagens do conjunto de dados mencionado. Lidamos, portanto, com um problema de **classificação**, que utiliza o *aprendizado supervisionado* como método de treinamento da rede neural. 

O modelo inicial contém 1.200.322 parâmetros treináveis, com três camadas convolucionais e duas camadas densamente conectadas. Os dados de entrada inicialmente consistem em 1000 amostras de cada categoria (infectadas e não infectadas), com resolução de 64x64x3, dos quais 20% foram separados para teste. Todas as camadas convolucionais possuem 32 neurônios, filtros 3x3, função de ativação **ReLU**, *pooling* de 2x2 e *dropout* de 0,2. O parâmetro *"same"* foi utilizado para preservar a dimensão espacial. Em seguida, há duas camadas densamente conectadas, com 512 e 256 neurônios, respectivamente, com função de ativação **ReLU**, seguidas por *dropout* com taxa de 0,2. O modelo foi configurado com o otimizador **ADAM** e treinado por 40 épocas. Na avaliação do modelo com os dados de teste, a acurácia alcançou **0.96**, indicando um desempenho satisfatório.

## Reprodução do modelo base

O modelo base foi reproduzido com 2000 imagens, desta vez em 100 épocas. 

```
history = model.fit(X_train, y_train, batch_size = 32, validation_split = 0.1, epochs = 100, verbose = 1)
```


Observou-se que a partir da 40ª época, a performance estabilizou, sem melhorias significativas.



A acurácia do modelo com os dados de teste foi de **0.95**, ligeiramente abaixo do índice base.

```
_,score = model.evaluate(X_test, y_test)
print(score)
13/13 [==============================] - 0s 4ms/step - loss: 0.4626 - accuracy: 0.9500
0.949999988079071
```




## Dataset completo

Reproduzimos o modelo base, com algumas modificações, utilizando o dataset completo, com
27.558 imagens. Para isto, baixamos o dataset diretamente do repositório disponível em:
https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html,
através do seguinte comando:

```
# Definir o endereço para baixar arquivo
!wget -P /content/drive/MyDrive/ELT579/Problema4 https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip
```




Extraímos o conteúdo do arquivo compactado no Google Drive, com o comando:

```
local_zip = '/content/drive/MyDrive/ELT579/Problema4/cell_images.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/drive/MyDrive/ELT579/Problema4')
zip_ref.close()
```



O modelo base foi reproduzido com algumas modificações, utilizando o dataset completo de 27.558 imagens. O número de neurônios na terceira camada convolucional foi aumentado para 64, enquanto as duas primeiras continuaram com 32. Foram adicionadas mais duas camadas convolucionais: a quarta com 64 neurônios e a quinta com 128. As características de pooling e dropout das novas camadas convolucionais permaneceram inalteradas. Os parâmetros treináveis foram reduzidos para 533.922.

O *dataset* foi separado em conjuntos de treinamento, validação e teste:

```
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.20, random_state=0)
```



O modelo foi ajustado em 100 épocas:

```
history = model.fit(X_train,
                    y_train,
                    batch_size = 32,
                    validation_split = 0.1,
                    epochs = 100, verbose = 1)
```



O modelo alcançou uma acurácia de **0.9459** com os dados de validação:

```
_,score = model.evaluate(X_val, y_val)
print(score)
138/138 [==============================] - 8s 61ms/step - loss: 0.1798 - accuracy: 0.9549
0.9548752903938293
```



Realizamos a predição com os dados de teste:

```
y_pred = model.predict(X_test)
y_pred = y_pred.astype(int).reshape(-1,)
y_test = y_test.astype(int).reshape(-1,)
```




Então, observamos uma queda considerável do
indicador, calculado através da função **classification_report:**

```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

     precision    recall  f1-score   support

           0       0.57      1.00      0.72      5512
           1       1.00      0.24      0.39      5512

    accuracy                           0.62     11024
   macro avg       0.78      0.62      0.56     11024
weighted avg       0.78      0.62      0.56     11024
```





A fração de imagens classificadas corretamente ficou em apenas **0.62**, índice muito baixo para o
tipo de problema estudado.

## Modelo *least_val_loss*

Um novo modelo definido como uma função foi adaptado do [notebook](https://colab.research.google.com/drive/16w3TDn_tAku17mum98EWTmjaLHAJcsk0?usp=sharing) de Kylie Ying - *Machine Learning for Everybody – Full Course* (https://www.youtube.com/watch?v=i_LwzRVP7bg). 

```
def train_model(X_train, y_train, conv_nodes, dense_nodes, dropout_prob, lr, batch_size, epochs):
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(conv_nodes, kernel_size = (3,3), activation = 'relu', padding = 'same',  input_shape=(64,64,3)),
      tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Conv2D(conv_nodes, kernel_size = (3,3), activation = 'relu', padding = 'same'),
      tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Conv2D(conv_nodes, kernel_size = (3,3), activation = 'relu', padding = 'same'),
      tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(dense_nodes, activation = 'relu'),
      tf.keras.layers.Dropout(rate = 0.2),
      tf.keras.layers.Dense(dense_nodes, activation = 'relu'),
      tf.keras.layers.Dropout(rate = 0.2),
      tf.keras.layers.Dense(2, activation= 'sigmoid'),
])

  nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=['accuracy'])
  history = nn_model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
  )

  return nn_model, history
```




Foram testados diferentes hiperparâmetros, e o modelo com o menor valor de função custo foi armazenado.:

```
least_val_loss = float('inf')
least_loss_model = None
epochs=100
for conv_nodes in [32, 64, 128]:
  for dense_nodes in [256, 512]:
    for dropout_prob in [0, 0.2]:
     for lr in [0.01, 0.001, 0.005]:
        for batch_size in [32, 64, 128]:
            print(f"conv nodes {conv_nodes}, dense nodes {dense_nodes}, fc_nodes dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
            model, history = train_model(X_train, y_train, conv_nodes, dense_nodes, dropout_prob, lr, batch_size, epochs)
            val_loss = model.evaluate(X_val, y_val)[0]
            if val_loss < least_val_loss:
              least_val_loss = val_loss
              least_loss_model = model
```




Salvamos e carregamos o modelo:

```
model.save('malaria_least_loss_model.h5')

from tensorflow.keras.models import load_model
model = load_model('malaria_least_loss_model.h5')
```



Ao avaliar o modelo com os dados de teste, obteve-se uma acurácia de **0.94**.

```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

              precision    recall  f1-score   support

           0       0.89      1.00      0.94      2993
           1       1.00      0.88      0.94      2993

    accuracy                           0.94      5986
   macro avg       0.94      0.94      0.94      5986
weighted avg       0.94      0.94      0.94      5986
```





## Transfer Learning: VGG16

Aqui construíremos um novo modelo, também com o *dataset* completo, utilizando a técnica de transferência de aprendizagem (*transfer learning*). Usaremos como base o modelo **VGG16** pré-treinado com o conjunto de dados *ImageNet*.

A ideia do *Transfer Learning* é usar o modelo pré-treinado para extrair algumas características dos nossos dados. Essa capacidade de extrair características foi previamente aprendida no *ImageNet*.

Para que o modelo seja treinado para o contexto dos nossos dados, excluímos as últimas camadas totalmente conectadas do **VGG16**, adicionando novas camadas para serem treinadas. Assim, podemos pensar que estamos apenas usando a rede pré-treinada para extrair características e treinando uma rede nova com essas características.

É importante que façamos com que os pesos das camadas extratoras da **VGG16** não se alterem durante o treinamento. Apenas as camadas totalmente conectadas que adicionaremos ao modelo é que serão alteradas durante o treinamento.

Essa abordagem é especialmente útil quando o conjunto de dados de destino é pequeno, em comparação com o conjunto de dados original no qual o *base_model* foi treinado.

Adaptamos o exemplo deste [notebook](https://colab.research.google.com/github/MatchLab-Imperial/deep-learning-course/blob/master/04_Common_CNN_architectures.ipynb#scrollTo=zwo-H2oSfQ0s).

Importamos as bibliotecas necessárias:

```
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, Sequential
from keras.layers import Lambda, Input
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator`
```



Carregamos o modelo **VGG16** sem as últimas camadas totalmente conectadas (include_top=False).
Redimensionamos as imagens de entrada para o tamanho esperado pelo modelo VGG16 e, em seguida, passamos essas imagens pelo modelo pré-treinado para obter as saídas correspondentes:

```
pre_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

newInput = Input(batch_shape=(None, 64, 64, 3))
resizedImg = Lambda(lambda image: tf.compat.v1.image.resize_images(image, (224, 224)))(newInput)
newOutputs = pre_model(resizedImg)
pre_model = Model(newInput, newOutputs)
```



Fazemos com que as camadas do modelo pré-treinado não sejam alteradas durante o treino:

```
for layer in pre_model.layers:
  layer.trainable = False
```



Criamos o modelo sequencial, com o **VGG16** acompanhado de novas camadas conectadas:

```
def define_model():
  model = Sequential()

  model.add(pre_model)

  model.add(Flatten())

  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.2))

  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.2))

  model.add(Dense(2, activation='sigmoid'))

  opt = Adam(learning_rate=0.001)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model
```



Apesar de inferior à metade do total, o número de parâmetros treináveis é o maior de todos os
modelos propostos até então, chegando a quase 13 milhões.

```
model = define_model()
model.summary()

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 model (Functional)          (None, 7, 7, 512)         14714688  
                                                                 
 flatten (Flatten)           (None, 25088)             0         
                                                                 
 dense (Dense)               (None, 512)               12845568  
                                                                 
 dropout (Dropout)           (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 256)               131328    
                                                                 
 dropout_1 (Dropout)         (None, 256)               0         
                                                                 
 dense_2 (Dense)             (None, 2)                 514       
                                                                 
=================================================================
Total params: 27692098 (105.64 MB)
Trainable params: 12977410 (49.50 MB)
Non-trainable params: 14714688 (56.13 MB)
```



Realizamos o ajuste do modelo em 50 épocas:

```
model.fit(X_train, y_train, batch_size=32, epochs=50)
```


Fizemos as predições com os dados de teste, foi obtida acurácia de **0.89**:

```
y_pred = model.predict(X_test)
y_pred = y_pred.astype(int).reshape(-1,)
y_test = y_test.astype(int).reshape(-1,)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

              precision    recall  f1-score   support

           0       0.82      1.00      0.90      2993
           1       1.00      0.78      0.88      2993

    accuracy                           0.89      5986
   macro avg       0.91      0.89      0.89      5986
weighted avg       0.91      0.89      0.89      5986
```



## Modelo K

O **Modelo K** utilizando, também baseado no método de *transfer learning*, foi adaptado de
Paulo Morillo (2020) - *“The transfer learning experience with VGG16 and Cifar 10 dataset”*,
[publicado](https://medium.com/analytics-vidhya/the-transfer-learning-experience-with-vgg16-and-cifar-10-dataset-9b25b306a23f#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjU1YzE4OGE4MzU0NmZjMTg4ZTUxNTc2YmE3MjgzNmUwNjAwZThiNzMiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMTIyMDEyMjQ4ODg4NjA2ODUyMTQiLCJlbWFpbCI6Imd1aWxoZXJtZWZlcnJlaXJhamZAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5iZiI6MTcwODc3ODIzOSwibmFtZSI6Ikd1aWxoZXJtZSBGZXJyZWlyYSIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NKR0o2bklmTTJ5NzFwc3FBTXhwc1VDVGYxYWtIb0JqYTdRaWxWMkVuV3lBTG89czk2LWMiLCJnaXZlbl9uYW1lIjoiR3VpbGhlcm1lIiwiZmFtaWx5X25hbWUiOiJGZXJyZWlyYSIsImxvY2FsZSI6InB0LUJSIiwiaWF0IjoxNzA4Nzc4NTM5LCJleHAiOjE3MDg3ODIxMzksImp0aSI6ImZkMWQyMTFkZDdmNDljNDE4NTUyMDkyYjYzMTc3YjA3ZTNlZWE0Y2MifQ.mT1984ehe7pi_As4EgmWT3871PGEpWgkACbrLt5zgpXyE76XK4semYcUh1A8QYGRmVasSU5dFzU0uUNS1WD4GCSFuYK74q8JQ7cJnmQWmVDr2-dj0yWCRUV3XU5FdxwuFFeZz-DxOzOsH1fQ1SD8i9AscZ6hZuOA9owZ0fphQQc_tc-Di-cpBB3zTBJWmUxlfuNgCzuOLL9LLQgi0QCWDPsSEJXfIumJx60dwN-3Y5yjZSbw567brt-cV65B5O4oSw6Pb0i46neQ-6HefvpKRaa-2KpgFz-Yz2hQJzPKnVWCynmkmrn6fEYLm3cqpVGdJW5Btu2m9NBkSEvF1b11OQ) em Analytics Vidhya, em 03/07/2020.

Realizamos o preprocessamento dos dados:

```
import numpy as np
import tensorflow
from tensorflow import keras as K
from sklearn.model_selection import train_test_split

def preprocess_data(X, Y):
    """ This method has the preprocess to train a model """
    X = X.astype('float32')
    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 2)  # Assuming binary classification, adjust accordingly
    return (X_p, Y_p)

if __name__ == "__main__":
    # Load your custom dataset
    dataset = np.array(dataset)
    label = np.array(label)

    # Divisão dos dados em treinamento, validação e teste

    X_train, X_temp, Y_train, Y_temp = train_test_split(dataset, label, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Split the dataset into training and validation sets
    #Xt, X, Yt, Y = train_test_split(dataset, label, test_size=0.2, random_state=42)

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_val_p, Y_val_p = preprocess_data(X_val, Y_val)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Now you can use Xt, Yt for training and X, Y for validation
    base_model = K.applications.vgg16.VGG16(include_top=False,
                                            weights='imagenet',
                                            pooling='avg')
```



A arquitetura foi definida de forma balanceada, incluindo o decaimento da taxa de aprendizagem, experimentações com diferentes resoluções e o armazenamento do melhor modelo:

```
model= K.Sequential()
model.add(K.layers.UpSampling2D())
model.add(base_model)
model.add(K.layers.Flatten())
model.add(K.layers.Dense(512, activation=('relu')))
model.add(K.layers.Dropout(0.2))
model.add(K.layers.Dense(256, activation=('relu')))
model.add(K.layers.Dropout(0.2))
model.add(K.layers.Dense(2, activation=('sigmoid')))
callback = []
def decay(epoch):
    """ This method create the alpha"""
    return 0.001 / (1 + 1 * 20)
callback += [K.callbacks.LearningRateScheduler(decay, verbose=1)]
callback += [K.callbacks.ModelCheckpoint('malaria_K2.h5',
                                          save_best_only=True,
                                          mode='min'
                                          )]
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=X_train_p, y=Y_train_p,
          batch_size=32,
          validation_data=(X_val_p, Y_val_p),
          epochs=20, shuffle=True,
          callbacks=callback,
          verbose=1
          )
```



Em problemas médicos, especialmente ao lidar com imagens microscópicas, é comum trabalhar com resoluções mais altas para capturar detalhes cruciais.
Por este motivo, foi utilizado o método *Upsampling2D*, para gerar mais pontos de dados de cada imagem do
banco de dados completo.



## Considerações finais

Ao avaliarmos o **Modelo K** com os dados de teste, obtivemos *acurácia* de **0.9846**:

```
score = model.evaluate(X_test_p, Y_test_p)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

47/47 [==============================] - 2s 39ms/step - loss: 0.0619 - accuracy: 0.9846
Test loss: 0.061920762062072754
Test accuracy: 0.9846359491348267
```



O base_model da **VGG16**, quando utilizado sem o argumento *input_shape*, assume um tamanho de entrada padrão que é (224, 224, 3). Isso se deve à arquitetura original do **VGG16** treinado no conjunto de dados *ImageNet*.

Para otimizar o desempenho e a acurácia, treinamos novamente o **Modelo K**, definido o parâmetro “input_shape”=(128,128,3)”:

```
    base_model = K.applications.vgg16.VGG16(include_top=False,
                                            weights='imagenet',
                                            pooling='avg',
                                            input_shape=(128, 128,3))
```



Então obtivemos uma *acurácia* de **0.9870** com os dados de teste, desempenho superior ao índice de referência e aos resultados de todos os modelos testados. Significa que, se o modelo faz 100 predições, acerta entre 98 e 99 delas. Portanto, a performance do modelo com os dados de teste pode ser considerada excelente.

```
score = model.evaluate(X_test_p, Y_test_p)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

94/94 [==============================] - 4s 40ms/step - loss: 0.2463 - accuracy: 0.9870
Test loss: 0.24626491963863373
Test accuracy: 0.9869695901870728
```




É importante destacar que usar o base_model da **VGG16** sem especificar o argumento *input_shape* pode resultar em perda de informações cruciais, principalmente ao lidar com conjuntos de dados que possuem dimensões de imagem variadas.

**Créditos**

*Este projeto é resultado das modificações feitas no modelo base desenvolvido pelo [Professor Sárvio Valente](https://sarvio.com.br/), da disciplina "Tópicos Especiais em Inteligência Artificial", do Curso de Pós-Graduação em Inteligência Artificial e Computacional da Universidade Federal de Viçosa. O objetivo é aprimorar as predições em um problema de classificação binária.*
