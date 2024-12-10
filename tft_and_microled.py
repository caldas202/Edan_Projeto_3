import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
from scipy.optimize import fsolve

plt.close('all')

Is2  =  5.0000e-16
vt2 =  1.3136
n2 =  1.5
Ut = 0.026
Imax=3e-6
print(Imax)
Vmax=n2*Ut*np.log(1+(Imax/Is2))+vt2
print(Vmax)
m = (Is2/(n2*Ut))*np.exp((Vmax-vt2)/(n2*Ut))
print(m)
b= Imax-m*Vmax
print(b)

caldas=[]
caldas=np.linspace(0, 3, 100) 

plt.figure(1)
def getlin_full(x):  # x = VGS;
    return np.piecewise(x,[x < vt2, (x >= vt2) & (x < Vmax), x >= Vmax],[lambda x: 0,lambda x: Is2 * (np.exp((x-vt2)/(n2*Ut)) - 1),lambda x: m*x+b])
new = getlin_full( caldas)
plt.plot(caldas, 1000 * new,label="ID(catodo-anodo)")
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA])")
plt.legend()  
plt.grid(True) 
plt.show()

VVV=getlin_full(2.15)

a = 1.0
VGS_max = 3
ID_on = 1e-6  

##########lambda
#parametros do tft new lambda
Ko = 9.095e-06
Vto = 0.322
mo = 1.9133

poly_fit = np.poly1d([0.000424621, -0.00385802, 0.0137971, -0.0154373, 0.0135855])        #define o polinomio a partir de coeficentes para poder colocar f(x)
def getlin_otim_w_l(x, c):  # x = VGS ; c = w/L 
    b=Ko/10
    return  (b*c) * np.power(np.maximum(x - Vto,0), mo)*(1 + ( poly_fit(x)* 0.85))



def error_function_1(w_l):
    VGS_test = 0.53
    ID_calculated = getlin_otim_w_l(VGS_test, w_l)
    return ID_calculated - ID_on

# Solve for w/l
w_l_solution_1 = fsolve(error_function_1, x0=1.0)  # x0 e um valor inicial possivel para w/L
print("w/l =", w_l_solution_1[0])

plt.figure(4)
new1 = getlin_otim_w_l(caldas,w_l_solution_1[0])
plt.plot(caldas, 1000* new1, label="ID(Vgs)")
plt.title("")
plt.xlabel("Voltage VGS [V]")
plt.ylabel("Current ID [mA]")
plt.legend()  
plt.grid(True) 
plt.show()

def ID_model_w_l(x, c): # x = VDS ; c = w/L
    b=Ko/10
    return np.piecewise(x,[x < a*(VGS_max-Vto), x >= a*(VGS_max-Vto)],
                        [lambda x: 2*((b*c)/a)*((VGS_max - Vto)**(mo-2))*(VGS_max - Vto - (x / (2 * a))) * x * (1 + ( poly_fit(VGS_max)* x)), 
                          lambda x: (b*c)*((VGS_max - Vto)**(mo))*(1 + ( poly_fit(VGS_max) * x))])

plt.figure(5)
new2 = ID_model_w_l(caldas,w_l_solution_1[0])
plt.plot(caldas, 1000 * new2,label="ID2(Vds)")
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA])")
plt.legend()  
plt.grid(True) 
plt.show()


########################

def ID_model_fim(x,y, c): # x = VDS ;y=Vgs c = w/L
    b=Ko/10
    return np.piecewise(y,[x < a*(y-Vto), x >= a*(y-Vto)],
                        [lambda y: 2*((b*c)/a)*np.power(np.maximum(y-Vto,0),mo-2)*(y - Vto - (x / (2 * a))) * x * (1 + ( poly_fit(y)* x)), 
                          lambda y: (b*c)*np.power(np.maximum(y-Vto,0),mo)*(1 + ( poly_fit(y) * x))])

print(getlin_otim_w_l(0.53,w_l_solution_1[0]))
print(ID_model_fim(0.85,0.53,w_l_solution_1[0]))

plt.figure(6)
new3 = ID_model_fim(3,caldas,w_l_solution_1[0])
plt.plot(caldas, 1000 * new3,label="ID2(Vgs)")
plt.plot(caldas, 1000 * new1,label="ID2(Vgs)")
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA])")
plt.legend()  
plt.grid(True) 
plt.show()
print(ID_model_fim(0.815/2,0.815/2,120))

# #parametros do tft antigo
# Ko = 9.095e-06
# Vto = 0.322
# mo = 1.9133
# lanbda = 0.02076
# a = 1.0
# VGS_max = 3
# ID_on = 1e-6  

# ##agora bem feito
# def getlin_otim_w_l(x, c):  # x = VGS ; c = w/L 
#     b=Ko/10
#     return  (b*c) * np.power(np.maximum(x - Vto,0), mo)*(1 + ( lanbda* 0.85))



# def error_function_1(w_l):
#     VGS_test = 0.53
#     ID_calculated = getlin_otim_w_l(VGS_test, w_l)
#     return ID_calculated - ID_on

# # Solve for w/l
# w_l_solution_1 = fsolve(error_function_1, x0=1.0)  # x0 e um valor inicial possivel para w/L
# print("w/l =", w_l_solution_1[0])

# plt.figure(2)
# new1 = getlin_otim_w_l(caldas,w_l_solution_1[0])
# plt.plot(caldas, 1000* new1, label="curva real ID(Vgs)",  linestyle='None',marker='.')
# plt.title("")
# plt.xlabel("Voltage VGS [V]")
# plt.ylabel("Current ID [mA]")
# plt.legend()  
# plt.grid(True) 
# plt.show()

# def ID_model_w_l(x, c): # x = VDS ; c = w/L
#     b=Ko/10
#     return np.piecewise(x,[x < a*(VGS_max-Vto), x >= a*(VGS_max-Vto)],
#                         [lambda x: 2*((b*c)/a)*((VGS_max - Vto)**(mo-2))*(VGS_max - Vto - (x / (2 * a))) * x * (1 + ( lanbda* x)), 
#                           lambda x: (b*c)*((VGS_max - Vto)**(mo))*(1 + ( lanbda * x))])

# plt.figure(3)
# new2 = ID_model_w_l(caldas,w_l_solution_1[0])
# plt.plot(caldas, 1000 * new2,label="ID2(Vds)")
# plt.title("")
# plt.xlabel("Voltage [V]")
# plt.ylabel("Current [mA])")
# plt.legend()  
# plt.grid(True) 
# plt.show()