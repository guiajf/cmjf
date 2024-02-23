# Malária

**Introdução**

Implementamos um modelo de rede neural convolucional para a detecção do
Plasmodium, parasita responsável pela transmissão da malária, em um conjunto de imagens de
células sanguíneas, mais especificamente das hemácias.

A malária é uma infecção dos glóbulos vermelhos do sangue causada por uma de cinco espécies
de protozoários Plasmodium.

O método diagnóstico largamente utilizado é o exame microscópico de amostras de sangue,
dependente da habilidade técnica do profissional.

Redes neurais convolucionais densamente conectadas podem ser utilizadas para extração de
características e classificação de imagens, para detecção do parasita responsável pela transmissão da
malária.

**Dataset**

O dataset contém 27.558 imagens de células, de forma balanceada, com igual quantidade de células
positivas e negativas, isto é, infectadas e não-infectadas. Disponibilizado pelo *National Library
of Medicine* dos Estados Unidos, pode ser baixado em:
https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip

**Indicadores de base**

O modelo inicial contém 1.200.322 parâmetros treináveis, possui
três camadas convolucionais e duas camadas densamente conectadas. Os dados de entrada são
constituídos inicialmente por 1000 amostras de cada uma das categorias (infectadas e não-
infectadas), com a resolução 64x64x3, dos quais foram separados 20% para teste.
Todas as camadas convolucionais possuem 32 neurônios,
utilizam filtros 3x3, função de ativação ReLU, pooling no formato 2x2 e dropout de 0,2. Foi
utilizado o argumento *“same”* como parâmetro da técnica de preenchimento ou *padding*, para
preservar a dimensão espacial. Em seguida, temos duas camadas densamente conectadas, com 512 e
256 neurônios respectivamente, com função de ativação **ReLU**, seguidos por *dropout* com taxa de
0,2. O modelo foi configurado com o otimizador **ADAM**, método de otimização baseado no
gradiente descendente estocástico, antes do treinamento em 40 épocas. Na etapa de avaliação do
modelo, com os dados de teste, foi alcançada a acurácia de **0,96**. Podemos afirmar que o modelo
performou muito bem na fase de teste, pois a fração das imagens que foram classificadas
corretamente pode ser considerada muito alta, para o tipo de problema abordado.

**Reprodução do modelo base**

Reproduzimos o modelo base, com 2000 imagens, dessa vez em 100 épocas:

![image](https://github.com/guiajf/malaria/assets/152413615/604ef65f-1fff-4f24-a584-f9aa62fe30e5)

Observamos que a partir de 40 épocas, a performance estabiliza, não apresenta melhoras.

![image](https://github.com/guiajf/malaria/assets/152413615/46930631-c7ec-4c46-b159-1d047fa679db)

A acurácia do modelo com os dados de teste ficou em 0,95, ligeiramente abaixo do índice base:

![image](https://github.com/guiajf/malaria/assets/152413615/3f4e5750-1921-4f0f-bd70-36480136be7a)


**Dataset completo**

Reproduzimos o modelo base, com algumas modificações, utilizando o dataset completo, com
27.558 imagens. Para isto, baixamos o dataset diretamente do repositório disponível em:
https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html,
através do seguinte comando:

![image](https://github.com/guiajf/malaria/assets/152413615/89560831-b615-451a-bb7a-de05a57be249)


Extraímos o conteúdo do arquivo compactado no Google Drive, com o comando:

![image](https://github.com/guiajf/malaria/assets/152413615/c0aca8ac-6d69-4969-b1c5-a2d61dc023e9)

Foram introduzidas as seguintes modificações: aumentamos o número de neurênios da terceira
camada convolucional, para 64, enquanto as duas primeiras continuaram com 32. Foram
adicionadas mais duas camadas convolucionais: a quarta com 64 neurônios, e a quinta com 128.
Permaneceram inalteradas as características de pooling e dropout das novas camadas
convolucionais. Os parâmetros treináveis foram reduzidos para 533.922, enquanto o modelo inicial
continha mais de 1,2 milhões.
Dessa vez, devido ao grande volume de dados disponíveis, separamos o conjunto de dados em
conjunto de treinamento, validação e teste:

![image](https://github.com/guiajf/malaria/assets/152413615/91b716a1-6533-418d-b162-3e105da0a2a0)

Efetuamos o ajuste desse modelo também em 100 épocas (constatamos posteriormente que foi
exagerado):

![image](https://github.com/guiajf/malaria/assets/152413615/b89cbf25-8e7e-4d50-8ffc-f91dd69b94b4)

A acurácia do modelo avaliado com os dados de validação alcançou 0,9459:

![image](https://github.com/guiajf/malaria/assets/152413615/dc7afbcb-254e-434f-93f2-298f85a51497)

Realizamos a predição com os dados de teste:

![image](https://github.com/guiajf/malaria/assets/152413615/1c9e608e-5ab9-4319-88fa-1f5f096e69c5)

Contudo, ao calcularmos a métrica de desempenho definida inicialmente no modelo, qual seja, a
*acurácia*, após a predição com os dados de teste, observamos que houve uma queda considerável do
indicador, calculado através da função **classification_report:**

![image](https://github.com/guiajf/malaria/assets/152413615/81fb2f7b-7a7a-4939-9ee6-03e34e20a544)

A fração de imagens classificadas corretamente ficou em apenas **0,62**, índice muito baixo para o
tipo de problema estudado.

**Modelo *least val loss***

Definimos um novo modelo como uma função, adaptado de Kylie Ying, 
em *Python TensorFlow for Machine Learning – Neural Network Text Classification Tutorial*:

Definimos um novo modelo como uma função:
![image](https://github.com/guiajf/malaria/assets/152413615/fce6fe72-7149-4469-b7ca-ff31ef4c177a)






