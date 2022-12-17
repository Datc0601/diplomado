import streamlit as st
import pandas as pd
import plotly.express as px
from email.policy import default
from operator import index
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import joblib
from sklearn import svm



@st.cache(persist = True)
def cargar_de_datos():
    leer_columnas = ['ESTU_DEPTO_RESIDE','ESTU_GENERO','FAMI_TIENECOMPUTADOR','FAMI_TIENEINTERNET','FAMI_ESTRATOVIVIENDA','FAMI_COMECARNEPESCADOHUEVO','PUNT_GLOBAL']
    df = pd.read_csv("icfes_data.csv",low_memory=False, usecols=leer_columnas)
    return df


st.set_page_config(
     page_title="PREDICCIÓN ICFES",
     layout="wide",
     initial_sidebar_state="expanded"
 )

df = cargar_de_datos().copy()
with st.sidebar: 
    st.image('logo.png')
    menu = option_menu(
        menu_title = "BIENVENIDO",
        options = ['Inicio','Exploración', 'Predicción','Contactenos'],
        default_index =0
    )


if menu == 'Inicio':

    # Título
    html_temp = """
    <h1 style="color:#618AE8;text-align:center;">ICFES COLOMBIA 2018-2021</h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    st.subheader("El dataset en el cual estamos trabajando es *Icfes Colombia 2018-2021* en este encontramos datos relacionados con las características de estudiantes que realizaron las  pruebas icfes en unos periodos determinados por año. En el dataset original se describe información   general, académica, familiar, socioeconómica y laboral de los estudiantes.")
    ver = st.button('Mostrar datos')
    if ver:
        st.dataframe(df)
        
        st.text("PUNTAJE GLOBAL POR DEPARTAMENTOS")
        cols = [i for i in df.columns if i.startswith('PUNT_')]
        inf = df.groupby(["ESTU_DEPTO_RESIDE"])[cols].mean()
        st.dataframe(inf)

        st.text("PUNTAJE GLOBAL - INTERNET")
        internet= df.groupby(["FAMI_TIENEINTERNET"])[["PUNT_GLOBAL"]].mean()
        st.dataframe(internet)

        st.text("PUNTAJE GLOBAL - COMPUTADOR")
        internet= df.groupby(["FAMI_TIENECOMPUTADOR"])[["PUNT_GLOBAL"]].mean()
        st.dataframe(internet)

        st.text("PUNTAJE GLOBAL POR ESTRATO")
        internet= df.groupby(["FAMI_ESTRATOVIVIENDA"])[["PUNT_GLOBAL"]].mean()
        st.dataframe(internet)

        st.text("PUNTAJE GLOBAL - ALIMENTACIÓN DEL ESTUDIANTE(CARNES-PESCADO-HUEVOS)")
        internet= df.groupby(["FAMI_COMECARNEPESCADOHUEVO"])[["PUNT_GLOBAL"]].mean()
        st.dataframe(internet)
    
elif menu == 'Exploración':

    # Título
    html_temp = """
    <h1 style="color:#618AE8;text-align:center;">Exploración realizada</h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    

    a1 = df.groupby(by=['ESTU_DEPTO_RESIDE']).mean()[['PUNT_GLOBAL']]
    
    
    fig1 = px.bar(
       
        a1,
        x = "PUNT_GLOBAL",
        y = a1.index,
        orientation = "h",
        height=700, width = 1200,
        title= "<b> PUNTAJE GLOBAL POR DEPARTAMENTOS </b>",
        color_discrete_sequence = ['#618AE8'] * len(a1),
        template = "plotly_white"
        ).update_layout(xaxis_title="PRORMEDIO DE PUNTAJE GLOBAL", yaxis_title="DEPARTAMENTOS",yaxis={'categoryorder':'total ascending'})
    
    
    fig1

    a2 = df.groupby(by=['FAMI_ESTRATOVIVIENDA']).mean()[['PUNT_GLOBAL']]
    
    
    fig2 = px.bar(
       
        a2,
        x = "PUNT_GLOBAL",
        y = a2.index,
        height=400, width = 1200,
        title= "<b> PROMEDIO DE PUNTAJE GLOBAL POR ESTRATOS </b>",
        color_discrete_sequence = ['#618AE8'] * len(a2),
        template = "plotly_white"
        ).update_layout(xaxis_title=" PUNTAJE DE PUNTAJE GLOBAL", yaxis_title="ESTRATO",yaxis={'categoryorder':'total ascending'})
    
    fig2

    a3 = df.groupby(by=['FAMI_COMECARNEPESCADOHUEVO']).mean()[['PUNT_GLOBAL']]
    fig3 = px.bar(
       
        a3,
        x = "PUNT_GLOBAL",
        y = a3.index,
        height=400, width = 1200,
        title= "<b> PUNTAJE GLOBAL POR ALIMENTACIÓN </b>",
        color_discrete_sequence = ['#618AE8'] * len(a3),
        template = "plotly_white"
        ).update_layout(xaxis_title="PROMEDIO DE PUNTAJE GLOBAL", yaxis_title="VECES QUE CONSUME CARNES-PESCADO-HUEVO",yaxis={'categoryorder':'total ascending'})
    
    fig3
    
    a3 = df.groupby(by=['ESTU_GENERO']).mean()[['PUNT_GLOBAL']]
    
    
    fig4 = px.bar(
       
        a3,
        x = "PUNT_GLOBAL",
        y = a3.index,
        orientation = "h",
        height=700, width = 1200,
        title= "<b> PUNTAJE GLOBAL POR GENERO </b>",
        color_discrete_sequence = ['#618AE8'] * len(a3),
        template = "plotly_white"
        ).update_layout(xaxis_title="PRORMEDIO DE PUNTAJE GLOBAL", yaxis_title="GENERO",yaxis={'categoryorder':'total ascending'})
    
    
    fig4

    def genero_est():
            genero= df['FAMI_TIENEINTERNET'].value_counts()

            datos = ['Si', 'No']
            colores = ["#618AE8","#5c60e6"]

            plt.pie(genero, labels=datos,autopct="%0.lf %%", colors= colores,radius=0.5)

            plt.title('INTERNET')
            plt.show()

            st.pyplot(plt)
    
    genero_est()
    
    
    

#PREDICCION

elif menu == 'Predicción':
   
    # Path del modelo preentrenado
    MODEL_PATH = 'my_model.pkl'


    # Se recibe la imagen y el modelo, devuelve la predicción
    def model_prediction(x_in, model):

        x = np.asarray(x_in).reshape(1,-1)
        preds=model.predict(x)

        return preds


    with open(MODEL_PATH, 'rb') as file:
        model = joblib.load(file)

    # Título
    html_temp = """
    <h1 style="color:#618AE8;text-align:center;">SISTEMA DE PREDICCIÓN APROBACIÓN PRUEBAS ICFES EN COLOMBIA</h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    st.subheader("Se pretende predecir si el estudiante gana o pierde las pruebas icfes teniendo en cuenta las variables socioeconomicas ")

    st.subheader(" Nota: Gana las pruebas icfes apartir de un puntacion mayor o igual a 250, de lo contrario pierde la prueba")

    estratos = {'Sin Estrato':0,'Estrato 1':1 , 'Estrato 2':2 ,'Estrato 3':3 , 'Estrato 4':4 ,'Estrato 5':5,'Estrato 6':6}

    genero = {'Femenino':0,'Masculino':1}

    internet = {'No':0,'Si':1}
    computador = {'No':0,'Si':1}
    
    N = st.selectbox("Genero" , ('Masculino','Femenino'))
    C = st.selectbox("Tiene Computador:", ('Si','No'))
    K = st.selectbox("Tiene Internet:", ('Si','No'))
    P = st.selectbox("Estrato de Vivienda:", ('Sin Estrato','Estrato 1','Estrato 2','Estrato 3','Estrato 4','Estrato 5','Estrato 6'))
    
    N = genero[N]
    P = estratos[P]
    K = internet[K]
    C = computador[C]
    
    
    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción"): 
        #x_in = list(np.float_((Datos.title().split('\t'))))
        x_in =[np.float_(N),
                    np.float_(P),
                    np.float_(K),
                    np.float_(C)
            ]
        predictS = model_prediction(x_in, model)
        
        pred = format(predictS[0]).upper() 
    
        if(int(pred) == 1):
            a = "PERDIO"
        else:
            a = "GANO"
            
        st.success('Se predice que el estudiante ' + a + " las pruebas ICFES")


elif menu == 'Contactenos':

        # Título
    html_temp = """
    <h1 style="color:#618AE8;text-align:center;">GRUPO 1</h1>
    <ul>
        <li>JORGE LUIS MONTES GOMEZ</li>
        <li>CUADRADO TRUJILLO DIEGO.</li>
        <li>PERALTA OSORIO LUIS</li>
        <li>DANIELA TORRES MINA</li>
        <li>DIAZ MONTIEL JEAN CARLOS</li>
        <li>MACEA PIMIENTA EDER</li>
    </ul>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    
       
         
    

    
    
    
