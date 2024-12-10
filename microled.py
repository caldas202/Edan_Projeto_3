import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.close('all')
Ut=0.026

def getlin_vt_is(x, a, b):  # x = VGS; a = Is; b = Vt
    return np.piecewise(x,[x < b, x >= b],[lambda x: 0, lambda x: a * (np.exp((x-b)/Ut) - 1)])
    #return a * (np.exp((x-b)/Ut) - 1)

def getlin_vt_is_n(x, a, b,c):  # x = VGS; a = Is; b = Vt; c = n
    return np.piecewise(x,[x < b, x >= b],[lambda x: 0, lambda x: a * (np.exp((x-b)/(c*Ut)) - 1)])
    #return a * (np.exp((x-b)/(c*Ut)) - 1)


    
plt.close('all')
dfVgs = pd.read_csv("C:/edgar/EDAN/trab3/microled.txt", sep=",", usecols=[0])
dfId = pd.read_csv("C:/edgar/EDAN/trab3/microled.txt", sep=",", usecols=[1])



[Is,vt], _ = curve_fit(getlin_vt_is, dfVgs.to_numpy()[:,0] , dfId.to_numpy()[:,0])
new = getlin_vt_is( dfVgs.to_numpy()[:,0] ,Is,vt)

plt.figure(1)
plt.plot(dfVgs, 1000 * dfId,linestyle='None',marker='.',label="ID(Vgs)")
plt.plot(dfVgs, 1000 * new,label="ID(Vgs)")
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA])")
plt.legend()  
plt.grid(True) 
#plt.show()
print("Is  = ",Is)
print("Vt = ",vt)


[Is2,vt2,n2], _ = curve_fit(getlin_vt_is_n,dfVgs.to_numpy()[:,0] , dfId.to_numpy()[:,0],bounds=([Is,0,0],[1e-15,2,5]),p0=[Is,vt,1.5])
new2 = getlin_vt_is_n( dfVgs.to_numpy()[:,0] ,Is2,vt2,n2)

plt.plot(dfVgs, 1000 * new2,label="ID2(Vgs)")
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA])")
plt.legend()  
plt.grid(True) 
print("Is2  = ",Is2)
print("Vt2 = ",vt2)
print("n2 = ",n2)

# def getlin_is_n(x, a,c):  # x = VGS; a = Is; b = Vt; c = n
#     return np.piecewise(x,[x < vt2, x >= vt2],[lambda x: 0, lambda x: a * (np.exp((x-vt2)/(c*Ut)) - 1)])
#     #return a * (np.exp((x-b)/(c*Ut)) - 1)
    
# [Is3,n3], _ = curve_fit(getlin_is_n,dfVgs.to_numpy()[:,0] , dfId.to_numpy()[:,0],bounds=([Is2*0.9,n2*0.9],[Is2*1.1,n2*1.1]),)
# new3 = getlin_is_n( dfVgs.to_numpy()[:,0] ,Is3,n3)

# plt.plot(dfVgs, 1000 * new3,label="ID2(Vgs)")
# plt.title("")
# plt.xlabel("Voltage [V]")
# plt.ylabel("Current [mA])")
# plt.legend()  
# plt.grid(True) 
# plt.show()
# print("Is3 = ",Is3)
# print("Vt3 = ",vt2)
# print("n3  = ",n3)


plt.figure(2)
Imax=3e-6
Vmax=n2*Ut*np.log(1+(Imax/Is2))+vt2

print("Vmax=",Vmax)
m = (Is2/(n2*Ut))*np.exp((Vmax-vt2)/(n2*Ut))

print("m=",m)
b= Imax-m*Vmax
print("b=",b)

caldas=[]
caldas=np.linspace(2, 3, 100) 
def getlin_full(x):  # x = VGS;
    return np.piecewise(x,[x < vt2, (x >= vt2) & (x < Vmax), x >= Vmax],[lambda x: 0,lambda x: Is2 * (np.exp((x-vt2)/(n2*Ut)) - 1),lambda x: m*x+b])

new4 = getlin_full( caldas)

plt.plot(caldas, 1000 * new4,label="ID2(Vgs)")
plt.title("")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA])")
plt.legend()  
plt.grid(True) 
plt.show()





    #return a * (np.exp((x-b)/(c*Ut)) - 1)

# dfVgs = pd.read_csv("C:/edgar/EDAN/trab3/microled.txt", sep=",", usecols=[0])
# dfId = pd.read_csv("C:/edgar/EDAN/trab3/microled.txt", sep=",", usecols=[1])

# def getlin_n(x,c):  # x = VGS; a = Is; b = Vt; c = n
#     return np.piecewise(x,[x < vt, x >= vt],[lambda x: 0, lambda x: Is * (np.exp((x-vt)/(c*Ut)) - 1)])
#     #return a * (np.exp((x-b)/(c*Ut)) - 1)

# [n2], _ = curve_fit(getlin_n,dfVgs.to_numpy()[:,0] , dfId.to_numpy()[:,0],bounds=([0.99],[1]))
# new2 = getlin_n( dfVgs.to_numpy()[:,0] ,n2)

# plt.plot(dfVgs, 1000 * new2,label="ID2(Vgs)")
# plt.title("")
# plt.xlabel("Voltage [V]")
# plt.ylabel("Current [mA])")
# plt.legend()  
# plt.grid(True) 
# plt.show()
# print("n2 = ",n2)
