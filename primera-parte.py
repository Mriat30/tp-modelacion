import math 
import numpy as np
from matplotlib import pyplot as plt

#LA PARTE A) DE ESTA PRIMERA PARTE FUE REALIZADA EN HOJA. LEER INFORME PARA MÁS DETALLES.

#--------------------------- ITEM (B) ---------------------------
#Mediciones EXACTAS
P1 = 106716
P2 = 106031
P = (P1 + P2) / 2
x0 = -P/3500
x1 = P/3500
y0 = y1 = P/2100
L = P/1300


#Por Newton-Raphson, hallamos mu despejando la misma de la ecuación 4), dado que en este caso se cumplen las condiciones
def f(mu):
    return ((np.sinh(mu * x1) - np.sinh(mu * x0)) / mu) - L

# Derivada de f(mu)
def df(mu):
    return (np.sinh(mu*x0) - (mu*x0*np.cosh(mu*x0)) - (np.sinh(mu*x1)) + (mu*x1*np.cosh(mu*x1))) / mu**2

#((mu * (x1 * np.cosh(mu * x1) - x0 * np.cosh(mu * x0))) - (np.sinh(mu * x1) - np.sinh(mu * x0))) / (mu ** 2)

tolerancia = 0.5e-6
# Newton-Raphson loop
def newton_raphson(semilla, tolerancia, NMAX):
    # Valor inicial para mu y tolerancia
    iteration = 0
    mu = semilla
    es_mayor_que_tolerancia = True
    while es_mayor_que_tolerancia and iteration < NMAX:
        mu_new = mu - f(mu) / df(mu)
        if abs(mu_new - mu) < tolerancia:
            es_mayor_que_tolerancia = False
        mu = mu_new
        iteration += 1
    return mu, iteration

mu, iteration = newton_raphson(0.1, tolerancia, 100)
print(mu, iteration)
mu = float(f"{mu:.5g}") #para redondear a 6 decimales significativos

#Cálculo del C2
C2 = y1*mu - (np.cosh(mu*x1)) #despejando de la ecuación (3)

#propagación de errores c2 -> mu tiene error inherente para su cálculo

dc2_mu = y1 - (x1*np.sinh(mu*x1)) #derivada respecto a mu de C2
ec2 = float(f"{(abs(dc2_mu) * abs(tolerancia)):.1g}" ) #cota de C2 ya redondeada
t = math.ceil(-math.log10(2 * ec2)) #decimales significativos

C2 = float(f"{C2:.{t}}")

#para la ecuacion original, mu y C2 como datos de entrada tienen error inherente
def catenaria(x):
    return ((np.cosh(mu*x))/mu) + C2

def df_catenaria_mu(x):
    return ((x*mu*np.sinh(mu*x)) - np.cosh(mu*x))/mu**2

#y la derivada respecto de C2 es 1 -> se propaga su error como es de entrada
y_x_0 = catenaria(0)
e_y = (abs(df_catenaria_mu(0)) * tolerancia) + 1*ec2
t = math.ceil(-math.log10(2 * e_y))
y_x_0 = float(f"{y_x_0:.{t}}")

print("------------- ITEM B) -------------")
print("Valor de mu: ", mu)
print("Valor de C2: ", C2)
print("Valor de y(x=0): ", y_x_0)


#--------------------------- ITEM (C) ---------------------------
#A priori, los cálculos serían los mismos. Solo se modifican sus errores
#f(mu) = 2senh(mu * x1)/mu - L
print("------------- ITEM C) -------------")
COTA_X0 = COTA_X1 = COTA_Y0 = COTA_Y1 = COTA_L = 0.05

def cpexp(mu, delta_mu): 
    return abs(f(mu*(1+delta_mu))-f(mu))/(abs(f(mu))*abs(delta_mu))

def cota_error_propagado_mu(mu_medido, x1_medido, x0_medido):
    dphi_x1 = -((np.cosh(x1_medido * mu_medido)*df(mu_medido)) - f(mu_medido)*x1_medido*np.sinh(x1_medido*mu_medido))/((df(mu_medido)) ** 2)
    dphi_x0 = -(-np.cosh(mu_medido*x0_medido) + x0_medido*np.sinh(x0_medido*mu_medido)*f(mu_medido))/((df(mu_medido)) ** 2)
    dphi_l = 1/(df(mu_medido))
    cota = COTA_X1

    return cota * (abs(dphi_x1) + abs(dphi_x0)+ abs(dphi_l))


mu_n_menos_1, temp = newton_raphson(0.1, tolerancia, iteration-1)
cota_error_mu = cota_error_propagado_mu(mu_n_menos_1, x1, x0) + tolerancia #error total = error inh + error trunc (se desprecian err redondeo)
t = math.ceil(-math.log10(2 * cota_error_mu)) #decimales significativos
cota_error_mu = round(cota_error_mu, t)
perturbaciones = []
cps_exp = []
for i in range(1,16):
    delta_mu = 0.1**i
    perturbaciones.append(delta_mu)
    cps_exp.append(cpexp(mu, delta_mu))

perturbaciones = np.ravel(perturbaciones)
cps_exp = np.ravel(cps_exp)
print("Cota de error propagado de mu (CP analítico):", cota_error_mu)
plt.figure(figsize=(10, 6))
plt.scatter(perturbaciones, cps_exp)
plt.xlabel('Perturbacion delta_mu')
plt.ylabel('CP mu')
plt.title('Condición del problema para mu, en base a perturbaciones')
plt.grid(True)
plt.show()
mu_redondeado = round(mu, t-1)
print("Valor de mu bien redeondeado:", mu_redondeado)

#c2 = f(mu, x1, y1) = y1 - cosh(mu * x1)/mu

def cota_error_total_c2(mu_medido, x1_medido):
    df_mu = (np.cosh(x1_medido*mu_medido) - (mu_medido*x1_medido*np.sinh(mu_medido*x1_medido))) / mu_medido**2
    df_x1 = - mu_medido*np.sinh(mu_medido * x1_medido)
    return abs(cota_error_mu*df_mu) + abs(COTA_X1*df_x1) + abs(COTA_Y1 * 1)

cota_error_c2 = cota_error_total_c2(mu_redondeado, x1)
cota_error_c2 = round(cota_error_c2, 1)
print("Cota de error de C2(redondeado a 1 cifra):",cota_error_c2)
#erc2 = (cota_error_c2/C2)*100 -> Por si se desea obtener error relativo de C2. Da alrededor de 54%
print("C2 no puede expresarse bien redondeado (el error relativo es mayor al 50%), por ende lo expresamos como cota:", C2,"±", cota_error_c2)

#f(mu_medido, x_medido, c2_medido) =  cosh(mu * x)/mu + C2
#el error inherente de mu (dato de entrada para esta ecuación) es el obtenido anteriormente, por lo que se reutiliza esa variable
def cota_total_error_y(mu_medido, x_medido):
    df_mu = df_catenaria_mu(x_medido)
    df_x = mu_medido*np.sinh(mu_medido * x_medido)
    df_c2 = 1
    return abs(cota_error_mu*df_mu) + abs(cota_error_c2 * df_c2) + abs(COTA_X1*df_x)

cota_error_y = round(cota_total_error_y(mu_redondeado, 0), 1)
print("Cota de error de y(x) (redondeado a 1 cifra):", cota_error_y)
print("Valor de y(0) bien redeondeado:", round(y_x_0))
print("El valor expresado como intervalo", y_x_0, "±", cota_error_y)