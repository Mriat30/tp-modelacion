import numpy as np

# Definimos las constantes según nuestro caso del modelo
x0 = -7.35  # Asumiendo que x0 = -x1
x1 = 7.35   # Asumiendo que x1 = -x0
L = 49   # Longitud de la cadena. Realmente mide 50cm, pero se consideró el espacio perdido en el soporte del primer eslabón de cada lado
n = 20  #Puntos entre el intervalo

# Definimos la función f(mu) basada en la condición de longitud
def f(mu):
    return (np.sinh(mu * x1) - np.sinh(mu * x0)) / mu - L

# Derivada de f(mu)
def df(mu):
    return ((mu * (x1 * np.cosh(mu * x1) - x0 * np.cosh(mu * x0))) - 
            (np.sinh(mu * x1) - np.sinh(mu * x0))) / (mu ** 2)

# Valor inicial para mu y tolerancia
mu = 0.1  # Semilla inicial (puede ajustarse este valor)
tolerance = 0.5e-4 #Comparado al error inherente tomado (espesor) se considera casi despreciable

# Newton-Raphson loop
iteration = 0
while True:
    mu_new = mu - f(mu) / df(mu)
    if abs(mu_new - mu) < tolerance:
        break
    mu = mu_new
    iteration += 1

#Cálculo del C2
y1 = 24.0 #altura del soporte de la derecha (igual a y0)
C2 = y1*mu - (np.cosh(mu*x1)) #despejando de la ecuación (3)


# Resultados
print("Número de iteraciones:", iteration)
print("Valor de mu:", mu)
print("Valor de C2", C2)


#-------------- Ajuste --------------

puntos = np.array([-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, 2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
soluciones_foto = np.array([
    11.464, 8.760, 6.519, 4.898, 3.809, 2.975, 2.309, 1.815, 1.469, 1.241,
    1.119, 1.278, 1.525, 1.846, 2.343, 2.911, 3.761, 4.593, 5.880, 7.287
])


def catenaria(x):
    return ((np.cosh((mu*x)))/mu + C2)

def obtener_soluciones_catenaria():
    y = []
    for punto in puntos:
        y.append(catenaria(punto))

    return y

soluciones_catenaria = np.matrix(obtener_soluciones_catenaria())
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

b=(np.inner(fi0, soluciones_foto), np.inner(fi1, soluciones_foto), np.inner(fi2, soluciones_foto))
b=np.array(b).reshape(3,1)
b=np.matrix(b).reshape(3,1)

c=np.linalg.inv(M)*b

soluciones_cuadratica = c[0,0]*fi0+c[1,0]*fi1+c[2,0]*fi2
dif=soluciones_foto - soluciones_catenaria
ecm=np.sqrt(np.inner(dif, dif) / n)
print("Coeficientes del ajuste:", c) #Coeficientes de ajuste cuadrático c0*fi0 + c1*fi1 + c2*fi2 respectivamente
print("ECM Catenaria:", ecm[0,0])
dif=soluciones_foto - soluciones_cuadratica
ecm=np.sqrt(np.inner(dif, dif) / n)
print("ECM Cuadrática:", ecm[0,0])