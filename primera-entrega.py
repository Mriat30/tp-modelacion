import numpy as np

# Definimos las constantes según nuestro caso del modelo
x0 = -7.5  # Asumiendo que x0 = -x1
x1 = 7.5   # Asumiendo que x1 = -x0
cantidad_de_puntos = 22
L = 50   # Longitud entre los puntos (ajusta este valor según el problema)
n = 20

# Definimos la función f(mu) basada en la condición de longitud
def f(mu):
    return (np.sinh(mu * x1) - np.sinh(mu * x0)) / mu - L

# Derivada de f(mu)
def df(mu):
    return ((mu * (x1 * np.cosh(mu * x1) - x0 * np.cosh(mu * x0))) - 
            (np.sinh(mu * x1) - np.sinh(mu * x0))) / (mu ** 2)

# Valor inicial para mu y tolerancia
mu = 0.1  # Semilla inicial (puedes ajustar este valor)
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
y1 = 23.0 #altura del soporte de la derecha (igual a y0)
C2 = y1*mu - (np.cosh(mu*x1)) #despejando de la ecuación (3)


# Resultados
print("Valor de mu:", mu)
print("Número de iteraciones:", iteration)
print("Valor de C2", C2)


#-------------- LINEALIZACION --------------

puntos = np.linspace(x0 + (x1 - x0) / (n+1), x1 - (x1 - x0) / (n+1), n)
#print("puntos",puntos)

def catenaria(x):
    return ((np.cosh((mu*x)))/mu + C2)

def obtener_soluciones_catenaria():
    y = []
    for punto in puntos:
        y.append(catenaria(punto))

    return y

soluciones = np.matrix(obtener_soluciones_catenaria())
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

b=(np.inner(fi0, soluciones), np.inner(fi1, soluciones), np.inner(fi2, soluciones))
b=np.array(b).reshape(3,1)
b=np.matrix(b).reshape(3,1)

c=np.linalg.inv(M)*b

soluciones_medidas = c[0,0]*fi0+c[1,0]*fi1+c[2,0]*fi2
dif=soluciones - soluciones_medidas
ecm=np.sqrt(np.inner(dif, dif) / n)
print("ECM", ecm[0,0])

#print("soluciones", soluciones)