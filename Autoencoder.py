import numpy as np

figurita = np.array ([
    [255, 255, 255, 255, 255, 255, 255, 255],
    [255, 0, 1, 1, 1, 1, 0, 255],
    [255, 0, 255, 255, 255, 255, 0, 255],
    [255, 0, 253, 253, 253, 253, 0, 255],
    [255, 0, 254, 253, 253, 254, 0, 255],
    [255, 0, 255, 255, 255, 255, 0, 255],
    [255, 1, 1, 1, 1, 1, 1, 255],
    [255, 255, 255, 255, 255, 255, 255, 255]])

import numpy as np
import matplotlib.pyplot as plt

def sigmoide(z):
  return 1.0/(1.0+(np.exp(-z)))

def deri_sig2(x):
  return np.exp(-x)/(1.0+np.exp(-x))**2

delta = 0.7
alpha=0.7
ECM= 3
epocas=20000
ecm_values = []
epoca= 0

np.random.seed(0)
X_r = np.random.randint(0, 256, size=( 8, 8))
X_g = np.random.randint(0, 256, size=(8, 8))
X_b = np.random.randint(0, 256, size=(8, 8))

X_rgb = np.array([X_r,X_g,X_b])
X_rgb = np.transpose(X_rgb, (1, 2, 0))
print(np.array([X_r,X_g,X_b]))
plt.imshow(X_rgb)
plt.title("PALETA DE COLORES")
plt.show()

X_rnormalizado= X_r/255
X_gnormalizado= X_g/255
X_bnormalizado= X_b/255

X_rn= X_rnormalizado.flatten()
X_gn= X_gnormalizado.flatten()
X_bn= X_bnormalizado.flatten()


X_rn = X_rn.reshape(1, -1)
X_gn = X_gn.reshape(1, -1)
X_bn = X_bn.reshape(1, -1)

X_normalizada = np.hstack([X_rn, X_gn, X_bn])
print(X_normalizada.shape)

neurona_entrada =  X_normalizada.shape[1]
neurona_latente = 1
neurona_salida = neurona_entrada

X = np.insert(X_normalizada,0,-1,axis=1)

W_entrada= 2*np.random.rand(X.shape[1],neurona_entrada) -1

W_oculta=2*np.random.rand(neurona_entrada+1,neurona_latente)-1

W_salida= 2*np.random.rand(neurona_latente+1,neurona_salida)-1



W_new_input= np.zeros(W_entrada.shape)
W_old_input= np.zeros(W_entrada.shape)

W_new_hidden= np.zeros(W_oculta.shape)
W_old_hidden= np.zeros(W_oculta.shape)

W_new_output= np.zeros(W_salida.shape)
W_old_output= np.zeros(W_salida.shape)



while(ECM >= 0.001 and epoca<=epocas):
     for j in range (0, X.shape[0]): #evalua casos

#for epoca in range (epocas):

        z_entrada = np.dot(X,W_entrada) #primera capa
        y_entrada = np.insert(sigmoide(z_entrada),0,-1,axis=1) #primera capa

        z_oculta = np.dot(y_entrada,W_oculta)
        y_oculta = np.insert(sigmoide(z_oculta),0,-1,axis=1)

        z_salida =  np.dot(y_oculta,W_salida)
        Yobt = sigmoide(z_salida)



        aux= (X[:, 1:]-Yobt)*deri_sig2(z_salida)
        aux2 = np.dot(aux,W_salida[1:, :].T)*deri_sig2(z_oculta)
        aux3 = np.dot(aux2,W_oculta[1:, :].T)*deri_sig2(z_entrada)

        W_salida = W_salida + delta*np.dot(y_oculta.T,aux)
        W_oculta = W_oculta + delta*np.dot(y_entrada.T,aux2)
        W_entrada = W_entrada + delta*np.dot(X.T,aux3)


#-------------------Actualizacion con Momentum------------------

        W_new_output = W_salida + alpha*(W_salida- W_old_output)+ delta*np.dot(y_oculta.T,aux)
        W_old_output = W_salida
        W_salida = W_new_output

        W_new_hidden = W_oculta + alpha*(W_oculta- W_old_hidden)+ delta*np.dot(y_entrada.T,aux2)
        W_old_hidden = W_oculta
        W_oculta = W_new_hidden

        W_new_input = W_entrada + alpha*(W_entrada-W_old_input)+ delta*np.dot(X.T,aux3)
        W_old_input = W_entrada
        W_entrada = W_new_input

     ECM = (1/2)*np.sum((X[:, 1:]-Yobt)**2)
     ecm_values.append(ECM)  # Almacenar el valor actual de ECM
     epoca = epoca + 1
     print("Epoca: ", epoca)
     print("Y obt de la epoca: ", Yobt)


     z_entrada = np.dot(X,W_entrada) #primera capa
     y_entrada = np.insert(sigmoide(z_entrada),0,-1,axis=1) #primera capa

     z_oculta = np.dot(y_entrada,W_oculta)
     y_oculta = np.insert(sigmoide(z_oculta),0,-1,axis=1)

     z_salida =  np.dot(y_oculta,W_salida)
     Yobt = sigmoide(z_salida)

# Graficar el ECM a lo largo de las épocas
plt.plot(range(1, epoca+1), ecm_values, marker='o')
plt.xlabel('Época')
plt.ylabel('Error Cuadrático Medio (ECM)')
plt.title('Evolución del ECM durante el entrenamiento')
plt.grid(True)
plt.show()
print("X: ", X.flatten())
print("Y obt final: ", np.round(Yobt*255))

forma_original = (3, 8, 8)
y= np.round(Yobt*255)
# Reorganizar Yobt para que tenga la forma original
Yobt_reshaped = y.reshape(forma_original)
print(Yobt_reshaped)

# Transponer Yobt_reshaped para cambiar el orden de las dimensiones
Yobt_transpuesta = Yobt_reshaped.transpose((2, 1, 0))/ 255.0

# Mostrar la imagen correspondiente a Yobt_transpuesta
plt.imshow(Yobt_transpuesta)
plt.title("Reconstruccion")
plt.show()

X_rgb_transpuesta = X_rgb.transpose((1, 0, 2))
plt.subplot(1, 2, 1)  # Primer subplot
plt.imshow(X_rgb_transpuesta)
plt.title("Original")

plt.subplot(1, 2, 2)  # Segundo subplot
plt.imshow(Yobt_transpuesta)
plt.title("Reconstruccion")

ruido = np.random.normal(loc=0, scale=20, size=X_rgb.shape)

# Aplicar el ruido a la imagen RGB
X_rgb_con_ruido = X_rgb + ruido

# Asegurar que los valores estén dentro del rango [0, 255]
X_rgb_con_ruido = np.clip(X_rgb_con_ruido, 0, 255)


plt.imshow(X_rgb_con_ruido.astype('uint8'))  # Convertir los valores a enteros sin signo de 8 bits
plt.title('Imagen con ruido')
plt.axis('off')  # Desactivar los ejes
plt.show()


# Normalizar los valores de la imagen con ruido
X_rgb_con_ruido_normalizado = X_rgb_con_ruido / 255.0

# Reformatear la imagen para que coincida con el formato de entrada del autoencoder
X_rgb_con_ruido_normalizado = X_rgb_con_ruido_normalizado.reshape(1, -1)

prueba = np.insert(X_rgb_con_ruido_normalizado,0,-1,axis=1)

z_entrada = np.dot(prueba,W_entrada) #primera capa
y_entrada = np.insert(sigmoide(z_entrada),0,-1,axis=1) #primera capa

z_oculta = np.dot(y_entrada,W_oculta)
y_oculta = np.insert(sigmoide(z_oculta),0,-1,axis=1)

z_salida =  np.dot(y_oculta,W_salida)
Yobt_prueba = sigmoide(z_salida)
print("X: ", prueba.flatten())
print("Y obt final: ", np.round(Yobt_prueba*255))

forma_original = (3, 8, 8)
y_prueba= np.round(Yobt_prueba*255)
# Reorganizar Yobt para que tenga la forma original
Yobt_reshaped_prueba = y_prueba.reshape(forma_original)
print(Yobt_reshaped_prueba)
# Transponer Yobt_reshaped para cambiar el orden de las dimensiones
Yobt_transpuesta_prueba = Yobt_reshaped.transpose((1, 2, 0))/ 255.0

# Mostrar la imagen correspondiente a Yobt_transpuesta
plt.imshow(Yobt_transpuesta_prueba)
plt.title("Reconstruccion")
plt.show()