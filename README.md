# Modelo de rede neural para diagnóstico da malária

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

Executamos o código em Python, no Google Colab, para reproduzir modelos de rede neural convolucional (CNN),
utilizando a API Keras do TensorFlow, capazes de classificar imagens do conjunto de dados mencionado.

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

![image](https://github.com/guiajf/malaria/assets/152413615/6ad0ab21-5aab-47f3-9786-c72650d9b48c)



A fração de imagens classificadas corretamente ficou em apenas **0,62**, índice muito baixo para o
tipo de problema estudado.

**Modelo *least val loss***

Definimos um novo modelo como uma função, adaptado de Kylie Ying, 
em *Machine Learning for Everybody – Full Course*, disponível em https://www.youtube.com/watch?v=i_LwzRVP7bg:

![image](https://github.com/guiajf/malaria/assets/152413615/fce6fe72-7149-4469-b7ca-ff31ef4c177a)

Testamos os melhores hiperparâmetros, a saber: o número de neurônios das camadas
convolucionais e densamente conectadas, a taxa de *dropout*, a taxa de aprendizagem e o *batch_size*.
E depois armazenamos o modelo que apresente o menor valor de função custo:

![image](https://github.com/guiajf/malaria/assets/152413615/303b895d-e5ab-4918-b46e-8b55a063f3a7)



Salvamos e carregamos o modelo:

![image](https://github.com/guiajf/malaria/assets/152413615/c693495e-b925-48f3-a9a6-0bec795087ac)

Ao avaliarmos o modelo com os dados de teste, alcançamos uma *acurácia* de **0.94**.

![image](https://github.com/guiajf/malaria/assets/152413615/6164b8a2-708a-44f7-ba9e-6cb05df2226b)



**Transfer Learning: VGG16**

Aqui construíremos um novo modelo, também com o *dataset* completo, utilizando a técnica de transferência de aprendizagem (*transfer learning*). Usaremos como base o modelo VGG16 pré-treinado com o conjunto de dados *ImageNet*.

A ideia do *Transfer Learning* é usar o modelo pré-treinado para extrair algumas características dos nossos dados. Essa capacidade de extrair características foi previamente aprendida no *ImageNet*.

Para que o modelo seja treinado para o contexto dos nossos dados, excluímos as últimas camadas totalmente conectadas do **VGG16**, adicionando novas camadas para serem treinadas. Assim, podemos pensar que estamos apenas usando a rede pré-treinada para extrair características e treinando uma rede nova com essas características.

É importante que façamos com que os pesos das camadas extratoras da **VGG16** não se alterem durante o treinamento. Apenas as camadas totalmente conectadas que adicionaremos ao modelo é que serão alteradas durante o treinamento.

Essa abordagem é especialmente útil quando o conjunto de dados de destino é pequeno, em comparação com o conjunto de dados original no qual o *base_model* foi treinado.

Carregamos as bibliotecas necessárias:

![image](https://github.com/guiajf/malaria/assets/152413615/a4d85130-9719-4531-b965-a45bb98f9b7d)

Carregamos o modelo VGG16 sem as últimas camadas totalmente conectadas (include_top=False):

![image](https://github.com/guiajf/malaria/assets/152413615/5801d7b5-9590-4e69-9d21-ed5855fed48a)

Fazemos com que as camadas do modelo pré-treinado não sejam alteradas durante o treino:

![image](https://github.com/guiajf/malaria/assets/152413615/29b6a1b0-2dea-4135-a27f-dcd7acb10bf0)

Criamos o modelo sequencial, com o **VGG16** acompanhado de novas camadas conectadas:

![image](https://github.com/guiajf/malaria/assets/152413615/05a91e41-d6b3-4cb4-b457-8f42ab631b07)

Apesar de inferior à metade do total, o número de parâmetros treináveis é o maior de todos os
modelos propostos até então, chegando a quase 13 milhões.

![image](https://github.com/guiajf/malaria/assets/152413615/7d8150ae-8ed3-42fb-b7ad-c5be46b7f9ff)

Realizamos o ajuste do modelo em 50 épocas:

![image](https://github.com/guiajf/malaria/assets/152413615/a37d330b-26c5-42dd-9cfa-613b3474ef65)

Fizemos as predições com os dados de teste, foi obtida acurácia de **0.89**:

![image](https://github.com/guiajf/malaria/assets/152413615/ed68a064-4d63-4ac3-9ee5-3e3d0a2f0ded)

**Modelo K**

O **Modelo Kutilizando**, também baseado no método de *transfer learning*, foi adaptado de
Paulo Morillo (2020) - *“The transfer learning experience with VGG16 and Cifar 10 dataset”*,
publicado em Analytics Vidhya, em 03/07/2020.

Realizamos o preprocessamento dos dados e definimos o modelo:

![image](https://github.com/guiajf/malaria/assets/152413615/091bfa5b-f150-41cd-9998-0c678f6a9c0a)

Em problemas médicos, especialmente ao lidar com imagens microscópicas, é comum trabalhar com resoluções mais altas para capturar detalhes cruciais.
Por este motivo, foi utilizado o método *Upsampling2D*, para gerar mais pontos de dados de cada imagem do
banco de dados completo.

Adicionar a camada *upsampling* pode ter implicações positivas e negativas.

**Positivas:**

**1. Aumento da Resolução Espacial:**

- O upsampling aumenta a resolução espacial dos dados de entrada. Isso pode ser benéfico se a resolução original das imagens do conjunto de dados for           relativamente baixa. Aumentar a resolução antes de passar pelos recursos aprendidos pelo *base_model* pode ajudar a preservar mais detalhes.

**2. Adaptação à Arquitetura do VGG16:**

- A arquitetura **VGG16**, pré-treinada no **ImageNet**, espera entradas com uma resolução específica (224x224 pixels). O *upsampling* pode ser usado para ajustar as dimensões das imagens do conjunto de dados para corresponder a essa resolução esperada.

**3. Exploração de Recursos Aprendidos:**

- O *upsampling* pode permitir que o modelo explore informações mais detalhadas nas imagens de entrada antes de passar por camadas convolucionais. Isso pode ser útil se as características de alta resolução forem relevantes para a tarefa.

**Negativas:**

**1. Aumento da Resolução:**

- O aumento da resolução das imagens pode fornecer mais detalhes e informações ao modelo. Isso pode ser benéfico para a tarefa específica, especialmente se detalhes finos são importantes.
  
**2. Overfitting Possível:**

Aumentar significativamente a resolução das imagens pode resultar em um modelo mais complexo e propenso ao overfitting, especialmente se o conjunto de dados original não possui imagens de alta resolução. O overfitting ocorre quando o modelo se ajusta demais aos dados de treinamento e não generaliza bem para novos dados.

**3. Dimensões Originais do dataset:**

O dataset contém imagens redimensionadas de 64x64 pixels. Ao redimensioná-las para 128x128 pixels, alteramos drasticamente as características do conjunto de dados original. Isso pode afetar a capacidade do modelo de generalizar para imagens com as dimensões originais.

**4. Uso de Recursos Computacionais:**

Aumentar a resolução também aumenta o custo computacional. Modelos maiores demandam mais recursos durante o treinamento e a inferência.

Definimos o decaimento da taxa de aprendizagem de acordo com as épocas.

Experimentamos com diferentes resoluções para encontrar um equilíbrio adequado.

A segunda chamada callback salva o melhor modelo, de acordo com o menor valor da função custo.

![image](https://github.com/guiajf/malaria/assets/152413615/30c4b04a-a25a-4ce6-88a1-27a923cbffa6)

Ao avaliarmos o **Modelo K** com om dados de teste, obtivemos *acurácia* de 0.9846:

![image](https://github.com/guiajf/malaria/assets/152413615/41e5dc51-d781-461a-886a-9e4e5b3f4ace)


O base_model da **VGG16**, quando utilizado sem o argumento *input_shape*, assume um tamanho de entrada padrão que é (224, 224, 3). Isso se deve à arquitetura original do VGG16 treinado no conjunto de dados *ImageNet*.

Quando o conjunto de dados tem dimensões de imagem diferentes, o *base_model* tenta lidar com essas diferenças automaticamente. Isso pode levar a uma perda de informações importantes, pois o modelo pode tentar redimensionar ou cortar as imagens para ajustá-las ao tamanho esperado.

Para otimizar o desempenho e a acurácia, treinamos novamente o **Modelo K**, definido o parâmetro “input_shape”=(128,128,3)”:

![image](https://github.com/guiajf/malaria/assets/152413615/4f63d711-851d-47d2-b55b-ca342f5167ab)

Então obtivemos uma *acurácia* de *0.9870* com os dados de teste.
Notamos que a alteração do argumento *input_shape* não provocou divergências significativas no desempenho dos modelos.

![image](https://github.com/guiajf/malaria/assets/152413615/ccba8a5f-e6dc-4b36-a17f-06e77e53cedb)























