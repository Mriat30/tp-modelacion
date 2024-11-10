import numpy as np
from matplotlib import pyplot as plt

#----------------- GENERALIDADES -----------------
#Las siguientes medidas y funciones valen para todos los casos del TPM (eg. todos los casos medimos la altura a 24cm)

L = 49   # Longitud de la cadena. Realmente mide 50cm, pero se consideró el espacio perdido en el soporte del primer eslabón de cada lado
n = 22  #Puntos considerando los extremos
y1 = y0 = 24.0 #altura del soporte

# Definimos la función f(mu) basada en la condición de longitud
def f(mu, x0, x1):
    return (np.sinh(mu * x1) - np.sinh(mu * x0)) / mu - L

# Derivada de f(mu)
def df(mu, x0, x1):
    return ((mu * (x1 * np.cosh(mu * x1) - x0 * np.cosh(mu * x0))) - 
            (np.sinh(mu * x1) - np.sinh(mu * x0))) / (mu ** 2)

# Newton-Raphson loop
tolerance = 0.5e-4
def NR(x0, x1):
    iteration = 0
    mu = 0.1  # Semilla inicial (puede ajustarse este valor)
    while True:
        mu_new = mu - f(mu, x0, x1) / df(mu, x0, x1)
        if abs(mu_new - mu) < tolerance:
            break
        mu = mu_new
        iteration += 1
    return mu, iteration

#ecuación original
def catenaria(x):
    return ((np.cosh((mu_raiz*x)))/mu_raiz + C2)

#auxiliar para obtener valores de la ecuación original en cada caso
def obtener_soluciones_catenaria(puntos):
    y = []
    for punto in puntos:
        y.append(catenaria(punto))

    return y

#función que muestra ajuste y comparaciones según el caso
def resultados_soluciones(puntos, soluciones_caso, titulo):
    soluciones_catenaria = np.matrix(obtener_soluciones_catenaria(puntos))
    puntos = np.matrix(puntos)

    fi0=np.matrix(np.ones(n))
    fi1=puntos
    fi2=np.power(puntos, 2)
    M=(
        (np.inner(fi0,fi0)), (np.inner(fi0,fi1)), (np.inner(fi0,fi2)),
        (np.inner(fi1,fi0)), (np.inner(fi1,fi1)), (np.inner(fi1,fi2)),
        (np.inner(fi2,fi0)), (np.inner(fi2,fi1)), (np.inner(fi2,fi2))
    )
    M=np.array(M).reshape((3,3))
    M=np.matrix(M).reshape((3,3))

    b=(np.inner(fi0, soluciones_caso), np.inner(fi1, soluciones_caso), np.inner(fi2, soluciones_caso))
    b=np.array(b).reshape(3,1)
    b=np.matrix(b).reshape(3,1)

    c=np.linalg.inv(M)*b

    soluciones_cuadratica = c[0,0]*fi0+c[1,0]*fi1+c[2,0]*fi2
    dif=soluciones_caso - soluciones_catenaria
    ecm=np.sqrt(np.inner(dif, dif) / n)
    print("Coeficientes del ajuste:", c) #Coeficientes de ajuste cuadrático c0*fi0 + c1*fi1 + c2*fi2 respectivamente
    print("ECM Catenaria:", ecm[0,0])
    dif=soluciones_caso - soluciones_cuadratica
    ecm=np.sqrt(np.inner(dif, dif) / n)
    print("ECM Cuadrática:", ecm[0,0])

    puntos = np.ravel(puntos)
    soluciones_catenaria = np.ravel(soluciones_catenaria)
    soluciones_cuadratica = np.ravel(soluciones_cuadratica)

    # Gráficos para comparar ajuste, catenaria y datos medidos
    plt.figure(figsize=(10, 6))
    plt.scatter(puntos, soluciones_caso, color='blue', label='Datos Medidos')
    plt.plot(puntos, soluciones_catenaria, color='green', linestyle='-', label='Catenaria')
    plt.plot(puntos, soluciones_cuadratica, color='red', linestyle='--', label='Ajuste Cuadrático')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.show()


#----------------- CASO 0.8L -----------------
print("----------------- CASO 0.8L -----------------")
x0 = -19.6
x1 = 19.6

puntos = np.array([x0, -15.0, -13.5, -12.0, -10.5, -9.0, -7.5, -6.0, -4.5, -3.0, -1.5, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, x1])
soluciones_foto = np.array([
    y0,
    18.864, 17.129, 15.749, 14.508, 13.355, 12.468, 11.722, 11.203, 10.836, 10.659,
    10.635, 10.826, 11.256, 11.775, 12.498, 13.387, 14.313, 15.467, 16.826, 18.397,
    y1
])

#calculo de mu y c2
mu_raiz, iteraciones = NR(x0, x1)
C2 = y1*mu_raiz - (np.cosh(mu_raiz*x1)) #despejando de la ecuación (3)

# Resultados
print("Número de iteraciones:", iteraciones)
print("Valor de mu:", mu_raiz)
print("Valor de C2", C2)
resultados_soluciones(puntos, soluciones_foto, 'Caso: Comparación de funciones para 0.8L')


#CASO 0.7L
print("----------------- CASO 0.7L -----------------")
x0 = -17.15
x1 = 17.15
puntos = np.array([x0, -15.0, -13.5, -12.0, -10.5, -9.0, -7.5, -6.0, -4.5, -3.0, -1.5, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, x1])
soluciones_foto = np.array([
    y0,
    20.951, 17.932, 15.504, 13.615, 11.892, 10.566, 9.522, 8.775, 8.254, 7.944,
    7.972, 8.311, 8.846, 9.636, 10.623, 11.952, 13.559, 15.560, 17.846, 20.413,
    y1
])

#calculo de mu y c2
mu_raiz, iteraciones = NR(x0, x1)
C2 = y1*mu_raiz - (np.cosh(mu_raiz*x1)) #despejando de la ecuación (3)

# Resultados
print("Número de iteraciones:", iteraciones)
print("Valor de mu:", mu_raiz)
print("Valor de C2", C2)
resultados_soluciones(puntos, soluciones_foto, 'Caso: Comparación de funciones para 0.7L')

#----------------- CASO 0.3L -----------------

# Definimos las constantes según nuestro caso del modelo
print("----------------- CASO 0.3L -----------------")
x0 = -7.35
x1 = 7.35
puntos = np.array([x0, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, x1])
soluciones_foto = np.array([
    y0,
    11.464, 8.760, 6.519, 4.898, 3.809, 2.975, 2.309, 1.815, 1.469, 1.241,
    1.119, 1.278, 1.525, 1.846, 2.343, 2.911, 3.761, 4.593, 5.880, 7.287,
    y1
])

#calculo de mu y c2
mu_raiz, iteraciones = NR(x0, x1)
C2 = y1*mu_raiz - (np.cosh(mu_raiz*x1)) #despejando de la ecuación (3)

# Resultados
print("Número de iteraciones:", iteraciones)
print("Valor de mu:", mu_raiz)
print("Valor de C2", C2)
resultados_soluciones(puntos, soluciones_foto, 'Caso: Comparación de funciones para 0.3L')