#Importação das bibliotecas
import PySimpleGUI as sg
import cv2
import numpy as np
import tensorflow as tf

#Carregamento do modelo treinado
modelo = tf.keras.models.load_model('newOCR2048-512.h5')


class TelaPython:
    def __init__(self):
        global classei
        #Definição do layout da tela
        sg.theme('Reddit')
        layout = [[sg.Image(filename='', key='image')],
                  [sg.Text(size=(24, 1), key='Classe', font='Arial 22'), sg.Text(size=(12, 1), key='Probabilidade', font='Arial 22')],
                  [sg.Button('Ok'), sg.Button('Cancel')]]
        #Exibição da janela
        janela = sg.Window('IHM - OCR', layout, location=(400, 100))

        #Realização da captura da câmera
        cap = cv2.VideoCapture(0)

        #Captura da imagem convertendo a um frame.
        _, frame = cap.read()

        while True:
            #Tratamento da imagem visualizada
            event, values = janela.Read(timeout=20, timeout_key='timeout')
            _, frame = cap.read()
            img = np.asarray(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = img / 255
            img = cv2.resize(img, (64, 64))
            img = img.reshape(1, 64, 64, 1)
            #Previsão da letra de acordo com o modelo
            previsao = modelo.predict(img)
            classe = np.argmax(previsao[0])
            #Conversão em probabilidade
            probabilidade = np.amax(previsao)
            probabilidade = float('{:.2f}'.format(probabilidade))*100
            probabilidade = str(probabilidade)
            #Conversão do valor do vetor para a letra em exibição
            if classe == 0: classei = '0'
            if classe == 1: classei = '1'
            if classe == 2: classei = '2'
            if classe == 3: classei = '3'
            if classe == 4: classei = '4'
            if classe == 5: classei = '5'
            if classe == 6: classei = '6'
            if classe == 7: classei = '7'
            if classe == 8: classei = '8'
            if classe == 9: classei = '9'
            if classe == 10: classei = 'A'
            if classe == 11: classei = 'B'
            if classe == 12: classei = 'C'
            if classe == 13: classei = 'D'
            if classe == 14: classei = 'E'
            if classe == 15: classei = 'F'
            if classe == 16: classei = 'G'
            if classe == 17: classei = 'H'
            if classe == 18: classei = 'I'
            if classe == 19: classei = 'J'
            if classe == 20: classei = 'K'
            if classe == 21: classei = 'L'
            if classe == 22: classei = 'M'
            if classe == 23: classei = 'N'
            if classe == 24: classei = 'O'
            if classe == 25: classei = 'P'
            if classe == 26: classei = 'Q'
            if classe == 27: classei = 'R'
            if classe == 28: classei = 'S'
            if classe == 29: classei = 'T'
            if classe == 30: classei = 'U'
            if classe == 31: classei = 'V'
            if classe == 32: classei = 'W'
            if classe == 33: classei = 'X'
            if classe == 34: classei = 'Y'
            if classe == 35: classei = 'Z'
            if classe == 36: classei = 'Sem Letra'
            #Escrita dos textos presente na tela
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            janela.find_element('image').Update(data=imgbytes)
            janela['Classe'].update('Classe prevista: ' + str(classei))
            janela['Probabilidade'].update('Probabilidade: ' + str(probabilidade) + '%')
            #Verifica se o botão Cancel é apertado para fechar a janela
            if event == sg.WINDOW_CLOSED or event == 'Cancel':
                break

tela = TelaPython()
