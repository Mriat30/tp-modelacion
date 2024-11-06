import numpy as np

# Definimos las constantes según nuestro caso del modelo
x0 = -7.5  # Asumiendo que x0 = -x1
x1 = 7.5   # Asumiendo que x1 = -x0
L = 15.0   # Longitud entre los puntos (ajusta este valor según el problema)

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