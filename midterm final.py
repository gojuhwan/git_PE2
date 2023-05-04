import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd

#--------------------Font설정-----------------------------------------
total_font_axis = {'weight': 'bold', 'size': 10}
total_font_title = {'weight': 'bold', 'size': 12}

font_axis = {'weight': 'bold', 'size': 12}
font_title = {'weight': 'bold', 'size': 18}
#---------------------------------------------------------------------
tree = ET.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")
root = tree.getroot()
# ------------------IV_graph-------------------------------------------
plt.subplot(2, 3, 4)

voltage_list, current_list=[], []
for i in root.iter():
    if i.tag == 'Voltage':
        voltage_list = list(map(float, i.text.split(',')))
    elif i.tag == 'Current':
        current_list = list(map(float, i.text.split(',')))

voltage = np.array(voltage_list)
current = np.abs(current_list)

afc = np.polyfit(voltage, current, 12)
af = np.polyval(afc,voltage)

plt.plot(voltage, af, 'r--', lw=2, label='best-fit')
plt.scatter(voltage, current, s=50, label='data')
#R_squared
R_squared = r2_score(current,af)
#한 점에서 Current값 출력
position_x, position_y=0.05,0.6
for x, y in zip([-2, -1, 1], [current[voltage == -2][0], current[voltage == -1][0], current[voltage == 1.0][0]]):
    if y < 1e-3:
        plt.text(position_x, position_y, f"{x}V: {y:.10e}A", transform=plt.gca().transAxes, fontsize=10)
    else:
        plt.text(position_x, position_y, f"{x}V: {y:.10f}A", transform=plt.gca().transAxes, fontsize=10)
    position_y-=0.1

plt.text(0.05, 0.7, f"R-squared: {R_squared:.20f}",
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round,pad=0.5'),    #
         fontsize=10,fontweight='bold')

plt.title('IV-analysis - with fitting', fontdict= total_font_title)
plt.xlabel('Voltage [V]', fontdict=total_font_axis)
plt.ylabel('Current [A]', fontdict=total_font_axis)
plt.yscale('logit')
plt.legend(loc='best')
plt.grid(True,axis='both', color='gray', alpha=0.5, linestyle='--')

# ----------------------Transmission_graph----------------------------------

plt.subplot(2, 3, 1)
plot_color = ['lightcoral', 'coral', 'gold', 'lightgreen', 'lightskyblue', 'plum', 'navy', 'black', 'red']
color_number = 0

wl, tm = [], []
DC_bias = -2.0
for i in root.iter():
    if i.tag == 'WavelengthSweep':
        if i.attrib.get('DCBias') == str(DC_bias):
            wl = list(map(float, i.find('L').text.split(',')))
            tm = list(map(float, i.find('IL').text.split(',')))
            plt.plot(wl, tm, plot_color[color_number], label=f'{DC_bias}V')
            DC_bias += 0.5
            color_number += 1
    elif i.tag == 'Modulator':
        if i.attrib.get('Name') == 'DCM_LMZC_ALIGN':
            wl = list(map(float, i.find('PortCombo').find('WavelengthSweep').find('L').text.split(',')))
            tm = list(map(float, i.find('PortCombo').find('WavelengthSweep').find('IL').text.split(',')))
            plt.plot(wl, tm, color='purple', linestyle=':')


plt.title('Transmission spectra - as measured', fontdict=total_font_title)
plt.xlabel('Wavelength [nm]', fontdict=total_font_axis)
plt.ylabel('Measured transmission [dB]', fontdict=total_font_axis)
plt.legend(ncol=3,loc='lower center', fontsize=9)
plt.grid(True,axis='both', color='gray', alpha=0.5, linestyle='--')

# --------------------------transmission_graph(R_spuared)------------------------------

plt.subplot(2, 3, 2)

import warnings
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

best_fit_list = []
r2=[]
for i in range(1, 9):
    afc = np.polyfit(wl, tm, i)
    af = np.polyval(afc, wl)
    R_squared = r2_score(tm, af)
    r2.append(R_squared)
    best_fit_list.append((i, af, R_squared))
    plt.plot(wl, af, plot_color[i], lw=2, label=f'{i}th')
    plt.scatter(wl, tm, s=10)

best_fit_list = sorted(best_fit_list, key=lambda x: abs(x[2] - 1))[:3]

position_x, position_y = 0.4, 0.5
for i, af, R_squared in best_fit_list:
    text_color = 'red' if R_squared == max([item[2] for item in best_fit_list]) else 'black'
    plt.text(position_x, position_y, f'Degree: {i}\nR_squared: {R_squared:.15f}',
             color=text_color,
             transform=plt.gca().transAxes,
             fontsize=8, fontweight='bold')
    position_y -= 0.1


plt.title('Transmission spectra - processed and fitting', fontdict=total_font_title)
plt.xlabel('Wavelength [nm]', fontdict=total_font_axis)
plt.ylabel('Measured transmission [dB]', fontdict=total_font_axis)
plt.legend(ncol=3,loc='lower center', fontsize=9)
plt.grid(True,axis='both', color='gray', alpha=0.5, linestyle='--')
#-------------------------Flat Transmission spectra--------------------------------------

afc = np.polyfit(wl,tm,8)
af = np.polyval(afc,wl)

plt.subplot(2,3,3)
color_number=0
DC_bias = -2.0
wl1,tm1 = [], []
for i in root.iter():
    if i.tag == 'WavelengthSweep':
        if i.attrib.get('DCBias') == str(DC_bias):
            wl1 = list(map(float, i.find('L').text.split(',')))
            tm1 = list(map(float, i.find('IL').text.split(',')))
            tm_flat = []  #ref 값을 뺸 transmission값
            for k in range(len(tm1)):
                a = tm1[k] - af[k]
                tm_flat.append(a)
            plt.plot(wl1, tm_flat, plot_color[color_number], label=f'{DC_bias}V')
            DC_bias += 0.5
            color_number += 1
ref_flat = [] #ref값도 평평하게 만들기
for k in range(len(tm)):
    a = tm[k] - af[k]
    ref_flat.append(a)
plt.plot(wl, ref_flat, color='r', linestyle=':')
plt.legend(ncol=3, loc='lower center', fontsize=10)
plt.title('Flat Transmission spectra - as measured', fontdict=font_title)  # 주석: 그래프 제목
plt.xlabel('Wavelength [nm]', fontdict=font_axis)  # 주석: x축 레이블
plt.ylabel('Measured transmission [dB]', fontdict=font_axis)  # 주석: y축 레이블
plt.grid(True,axis='both', color='gray', alpha=0.5, linestyle='--')  # 주석: 그리드 추가

plt.savefig('midterm final.png')
# plt.show()
#-------------------------------
def Data_csv():
    Lot, Wafer, Mask, TestSite, Name, Date, Operator, Row, Column, Analysis_Wavelength = [],[],[],[],[],[],[],[],[],[]
    for data in root.iter():
        if data.tag == 'OIOMeasurement':
            Date.append(data.get('CreationDate'))
            Operator.append(data.get('Operator'))

        elif data.tag == 'TestSiteInfo':
            Lot.append(data.get('Batch'))
            Column.append(data.get('DieColumn'))
            Row.append(data.get('DieRow'))
            TestSite.append(data.get('TestSite'))
            Wafer.append(data.get('Wafer'))
            Mask.append(data.get('Maskset'))

        elif data.tag == 'ModulatorSite':
            Name.append(data.find('Modulator').get('Name'))

        elif data.tag == 'DesignParameter':
            if data.attrib.get('Name') == 'Design wavelength':
                Analysis_Wavelength.append(data.text)


    return Lot, Wafer, Mask, TestSite, Name, Date, Operator, Row, Column, Analysis_Wavelength

Lot, Wafer, Mask, TestSite, Name, Date, Operator, Row, Column, Analysis_Wavelength = Data_csv()
# --------------------------------------------------------------------------------------------
def ScriptID(site):
    last_element = site[-1].split("_")[-1]
    return last_element.replace("C", "")

ScriptID(TestSite)
# --------------------------------------------------------------------------------------------
def ScriptVersion():
    with open('Script Version.txt', 'r+') as f:
        number = float(f.read())
        number += 0.1
        f.seek(0)  # 파일 포인터를 파일 시작점으로 이동
        f.write("{:.2f}".format(number))  # 파일 내용을 수정
        f.truncate()  # 파일 크기를 현재 위치로 자름
        print(number)
        return round(number,2)
# S_Version = ScriptVersion()
# --------------------------------------------------------------------------------------------
import os

user_profile = os.environ['USERPROFILE']
s = str(user_profile)[9:]

def print_num(var):
    num_dict = {'rhwng': 'Go', 'dbsmi': 'Yoon', 'jim26': 'Lee', '김교령': 'Kim'}
    return num_dict.get(var)
# --------------------------------------------------------------------------------------------

def d(a):
    b=max(a)
    if b >= 0.95:
        ErrorFlag = 0
        ErrorDescription = "No Error"
    else:
        ErrorFlag = 1
        ErrorDescription = "Ref. spec. Error"
    return ErrorFlag, ErrorDescription
ErrorFlag, ErrorDescription = d(r2)

Exel_data = pd.DataFrame({'Lot': Lot,'Wafer': Wafer, 'Mask': Mask, 'TestSite':TestSite, 'Name':Name, 'Date':Date,
                          'Script ID': ScriptID(TestSite),'Script Version': ScriptVersion(),'Script Owner':print_num(s),'Operator': Operator, 'Row': Row, 'Column': Column,
                          'ErrorFlag': ErrorFlag, 'Error description': ErrorDescription, 'Analysis Wavelength': Analysis_Wavelength,
                          'Rsq of Ref.spectrum (Nth)':max(r2), 'Max transmission of Ref. spec. (dB)': np.max(tm),
                          'Rsq of IV': R_squared, 'I at -1V [A]':current[4], 'I at -2V [A]':abs(current[-1])})


if not os.path.exists('AnalysisResult_C.csv'):
    Exel_data.to_csv('AnalysisResult_C.csv', index=False, mode='w', encoding='utf-8-sig')
else:
    Exel_data.to_csv('AnalysisResult_C.csv', index=False, mode='a', encoding='utf-8-sig', header=False)

