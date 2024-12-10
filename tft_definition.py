import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#def getlin(x, a, b, c):  # x = VGS; a = K; b = Vt; c = n
#    return np.piecewise(x,[x < b, x >= b],[lambda x: 0, lambda x: a * np.power(np.maximum(x - b, 0), c)])

def getlin_ID_GM(x, a, b):  # x = VGS; a = Vt; b = m;
    return np.piecewise(x,[x-a < 0, x-a >= 0],[lambda x: 0, lambda x: (x-a)/b])

#a) e b)
plt.close('all')
dfo = pd.read_csv("C:/E2/EDAN/P1G4_W50L5_Output.csv",skiprows=256, usecols=[0,1,2,3])
dfo_VG = dfo.iloc[:,2].values
dfo_VD = dfo.iloc[:,1].values
dfo_ID = dfo.iloc[:,3].values
df_sat= pd.read_csv("C:/E2/EDAN/P1G4_W50L5_Transfer_saturation.csv",skiprows=251, usecols=[0,1,2,3])
dfVG_sat = df_sat.iloc[:,1].values
dfVD_sat = df_sat.iloc[:,2].values
dfID_sat = df_sat.iloc[:,3].values
#dfGM_sat = df_sat.iloc[:,4].values
#c)
index = np.argmax(dfVG_sat)
dfVG_sat2 = dfVG_sat[: index+1]
dfVD_sat2 = dfVD_sat[: index+1]
dfID_sat2 = dfID_sat[: index+1]
#dfGM_sat2 = dfGM_sat[: index+1]



#########otimização
poly_fit = np.poly1d([0.000424621, -0.00385802, 0.0137971, -0.0154373, 0.0135855])        #define o polinomio a partir de coeficentes para poder colocar f(x)
lanbda = 0.02076
largura = (len(dfID_sat2)-1)

def getlin_otim(x, a, b, c):  # x = VGS; a = K, b = Vt, c = M;
    return  a * np.power(np.maximum(x - b, 0), c)*(1 + (lanbda* 7))
print(df_sat)
plt.figure(1)
[Ko,Vto,mo], _ = curve_fit(getlin_otim,dfVG_sat2[: largura].flatten(), dfID_sat2[: largura].flatten(),bounds=([0,0.31,1], [1,0.42,4]))
new6 = getlin_otim(dfVG_sat2[: largura], Ko,Vto,mo)
plt.plot(dfVG_sat2[: largura],1000* dfID_sat2[: largura], label="curva real ID(Vgs)",  linestyle='None',marker='.')
plt.plot(dfVG_sat2[: largura], 1000*new6, label="curva aproximada ID(Vgs)")     
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA]")
plt.legend()  
plt.grid(True) 
plt.show()
print("K0 =", Ko)
print("VTO =", Vto)
print("Mo =", mo)

########################para o trabalho 3 vgs=3

VGS_max = 3
indices = np.where(dfo_VG == VGS_max)[0] 
VDS_max = dfo_VD[indices]
ID_max = dfo_ID[indices]

#funcao id linear especial one
def ID_model2(x, a): # x = VDS, a = alpha        poly_fit(VGS_max)       ((VGS_max - Vt)**0) 
    return np.piecewise(x,[x < a*(VGS_max-Vto), x >= a*(VGS_max-Vto)],
                        [lambda x: 2*(Ko/a)*((VGS_max - Vto)**(mo-2))*(VGS_max - Vto - (x / (2 * a))) * x * (1 + (poly_fit(3)* x)), 
                          lambda x: (Ko)*((VGS_max - Vto)**(mo))*(1 + ( poly_fit(3)    * x))])
plt.figure(2)
[alpha], _ = curve_fit(ID_model2, VDS_max, ID_max,bounds=([0], [2]))
new8 = ID_model2(VDS_max, alpha)
plt.plot(VDS_max, 1000* ID_max, label="curva real ID(Vgs)",  linestyle='None',marker='.')
plt.plot(VDS_max, 1000*new8, label="curva aproximada ID(Vgs)")     
plt.title("")
plt.xlabel("Voltage VDS [V]")
plt.ylabel("Current ID [mA]")
plt.legend()  
plt.grid(True) 
plt.show()
print(f"Para VGS max = {VGS_max}:")
print("alpha", alpha)

########################para o trabalho 3 vds = 3

def getlin_otim2(x, a, b, c):  # x = VGS; a = K, b = Vt, c = M;
    return  a * np.power(np.maximum(x - b, 0), c)*(1 + ( poly_fit(x)* 1))

caldas=[]
caldas=np.linspace(0, 7, 100) 

plt.figure(3)
new9 = getlin_otim2(caldas, Ko,Vto,mo)
plt.plot(dfVG_sat2[: largura],1000* dfID_sat2[: largura], label="curva real ID(Vgs)",  linestyle='None',marker='.')
plt.plot(caldas,1000* new9, label="curva apx ID(Vgs) com Vds=0.815",  linestyle='None',marker='.')
plt.title("")
plt.xlabel("Voltage VGS [V]")
plt.ylabel("Current ID [mA]")
plt.legend()  
plt.grid(True) 
plt.show()

#nao sei
# plt.figure(4)
# newxx = ID_model2(VDS_max, 1)
# plt.plot(VDS_max, 1000* ID_max, label="curva real ID(Vgs)",  linestyle='None',marker='.')
# plt.plot(VDS_max, 1000*newxx, label="curva aproximada ID(Vgs)")     
# plt.title("")
# plt.xlabel("Voltage VDS [V]")
# plt.ylabel("Current ID [mA]")
# plt.legend()  
# plt.grid(True) 
# plt.show()