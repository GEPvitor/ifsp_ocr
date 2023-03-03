#Importação das bibliotecas
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout


pasta = 'myData' #Determina qual o nome da pasta com os dados
classe = [] #Cria uma lista para as classes
img = [] #Cria uma lista para as imagens
#timg = (224, 224, 3) #Determina as dimensões das imagens
lista = os.listdir(pasta) #Realiza a leitura das pastas
#print(lista)
print("Nr Total de Classes Detectadas:", len(lista)) #Mostra o numero de classes
nclasses = len(lista) #Atribui a lista às classes

for classei in range(0, nclasses): #Varre as classes
    listaimg = os.listdir(pasta + "/" + str(classei)) #Lista todas as pastas de imagens
    for x in listaimg: #Varre as imagens
        dir = pasta + '/' + str(classei) + '/' + x
        #print(dir)
        imgatual = cv2.imread(dir) #Lê todas as imagens
        imgatual = cv2.resize(imgatual, (64, 64)) #Iguala as dimensões das imagens
        img.append(imgatual) #Coloca a imagem atual na lista de imagens
        classe.append(classei) #Salva as pastas na lista de classes

print("Total de imagens na lista: ", len(img))
img = np.array(img)  # Atribui a lista de imagens à um array.
print("Total de IDs na lista de classes: ", len(classe))
classe = np.array(classe)  # Atribui a lista de classes à um array.

#Cria uma função de pré-processamento das imagens
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

#Organiza as imagens de acordo com as classes para o treinamento
Xtreino, Xteste, Ytreino, Yteste = train_test_split(img, classe)

#Realiza o pré-processamentos nas imagens
Xtreino= np.array(list(map(preProcessing, Xtreino)))
Xteste= np.array(list(map(preProcessing, Xteste)))

#Redimensiona as imagens
Xtreino = Xtreino.reshape(Xtreino.shape[0], 64, 64, 1)
Xteste = Xteste.reshape(Xteste.shape[0], 64, 64, 1)

#Categoriza as letras repetidas juntas.
Ytreino = np_utils.to_categorical(Ytreino, num_classes=37)
Yteste = np_utils.to_categorical(Yteste, num_classes=37)

#Inicia a criação do modelo
modelo = Sequential()
#Realiza uma convolução.
modelo.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(64, 64, 1), activation='relu'))
#Seleciona os maiores valores da convolução e retorna em matrizes.
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Dropout(0.5))
modelo.add(Flatten())
#Realiza uma convolução.
modelo.add(Conv2D(filters=96, kernel_size=(7, 7), activation='relu'))
#Seleciona os maiores valores da convolução e retorna em matrizes.
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Dropout(0.5))
modelo.add(Flatten())
#Monta uma rede de 2048 neurônios.
modelo.add(Dense(2048, activation='relu'))
#Monta uma rede de 512 neurônios.
modelo.add(Dense(512, activation='relu'))
#Monta uma rede de 37 neurônios.
modelo.add(Dense(37, activation='softmax'))


#Configura o modelo para o treinamento
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Realiza o treinamento do modelo.
hist = modelo.fit(Xtreino, Ytreino, validation_data=(Xteste, Yteste), epochs=10, shuffle=1, batch_size=128)

#Retorna os valores de pontos do treinamento.
loss, acc = modelo.evaluate(Xteste, Yteste, verbose=1)
print("Acurácia: {:5.2f}%".format(100 * acc))

#Salva o modelo treinado
modelo.save('NmodeloOCR1024-10e.h5')
