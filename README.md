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

![image](https://github.com/guiajf/malaria/assets/152413615/4f3c9156-cccb-4ffa-9541-3258a529c220)

Extraímos o conteúdo do arquivo compactado no Google Drive, com o comando:

![image](https://github.com/guiajf/malaria/assets/152413615/e4d0bb6c-4daa-41b5-ab0a-92618d158d16)





