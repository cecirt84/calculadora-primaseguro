import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from streamlit.runtime.scriptrunner import RerunException
import streamlit as st



def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        # fallback: hack visual sin errores
        st.session_state.update(_rerun_trigger=not st.session_state.get("_rerun_trigger", False))


# ---------- INICIALIZACIÓN GLOBAL (DEBE IR PRIMERO) ----------
if "pagina" not in st.session_state:
    st.session_state.update({
        "pagina": 1,
        "edad": 35,
        "experiencia": 2,
        "km": 50000,
        "antiguedad": 5,
        "siniestros": 0,
        "prima_modelo": 0.0,
        "multiplicador": 1.0,
        "prima_ajustada": 0.0,
        "mostro_prima": False,
        "inicializado": True
    })

# Ahora sí podemos configurar la página
st.set_page_config(
    page_title="Calculadora de Prima",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Y mostrar el progress bar
progress = st.session_state['pagina']/3
st.progress(progress)

# Configuración de estilos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def siguiente():
    st.session_state["pagina"] += 1

def anterior():
    st.session_state["pagina"] -= 1

# ---------- Carga de datos (asegúrate de tener estos archivos) ----------
xgb_model = joblib.load("xgb_model.joblib")
df_primabase = pd.read_csv("df_primabase.csv")
prov_riesgo = pd.read_csv("prov_riesgo.csv")
franja_riesgo = pd.read_csv("franja_riesgo.csv")
zona_riesgo = pd.read_csv("zona_riesgo.csv")
diatipo_riesgo = pd.read_csv("diatipo_riesgo.csv")

provincia_map = {
    1: "Araba/Álava", 2: "Albacete", 3: "Alicante/Alacant", 4: "Almería", 5: "Ávila", 6: "Badajoz",
    7: "Balears, Illes", 8: "Barcelona", 9: "Burgos", 10: "Cáceres", 11: "Cádiz", 12: "Castellón/Castelló",
    13: "Ciudad Real", 14: "Córdoba", 15: "Coruña, A", 16: "Cuenca", 17: "Girona", 18: "Granada", 19: "Guadalajara",
    20: "Gipuzkoa", 21: "Huelva", 22: "Huesca", 23: "Jaén", 24: "León", 25: "Lleida", 26: "Rioja, La", 27: "Lugo",
    28: "Madrid", 29: "Málaga", 30: "Murcia", 31: "Navarra", 32: "Ourense", 33: "Asturias", 34: "Palencia",
    35: "Palmas, Las", 36: "Pontevedra", 37: "Salamanca", 38: "Santa Cruz de Tenerife", 39: "Cantabria",
    40: "Segovia", 41: "Sevilla", 42: "Soria", 43: "Tarragona", 44: "Teruel", 45: "Toledo",
    46: "Valencia/València", 47: "Valladolid", 48: "Bizkaia", 49: "Zamora", 50: "Zaragoza", 51: "Ceuta", 52: "Melilla"
}
zona_map = {1: 'Interurbana', 2: 'Urbana'}


# ---------- Pestaña 1: Cálculo de prima base ----------
if st.session_state['pagina'] == 1:
    st.title(" 🚗 Predicción Inteligente de Prima")

    st.markdown("""
        <p style='font-size:16px; color:#cccccc; line-height:1.6; margin-bottom:2em'>
        Esta aplicación estima la <strong>prima base de tu seguro</strong> a partir del perfil del conductor y las características del vehículo, utilizando un modelo de Machine Learning. <br>
        Posteriormente, se aplica un <strong>ajuste por riesgo</strong> que toma en cuenta los hábitos de conducción.
        </p>""", unsafe_allow_html=True)

    st.markdown("___")
    col1, col_sep, col2 = st.columns([1, 0.05, 1])

    # --- Columna izquierda: Perfil del Conductor ---
    with col1:
        st.markdown("""
        <h6 style="
            display: flex; 
            align-items: center; 
            gap: 8px; 
            font-size: 1.1rem; 
            font-weight: 600; 
            color: #ffffff; 
            margin-bottom: 1rem;
        ">
            Perfil del Conductor
        </h6>
        """, unsafe_allow_html=True)

        st.session_state.edad = st.slider(
            "Edad del conductor",
            18, 75,
            st.session_state["edad"],
            1,
            key='edad_input'
        )

        max_exp = st.session_state.edad - 18

        if max_exp == 0:
            st.session_state.experiencia = 0
            st.markdown("ℹ️ A los 18 años se asume que no hay experiencia previa.")
        else:
            st.session_state.experiencia = st.slider(
                f"Años de experiencia (máx: {max_exp})",
                0, max_exp,
                st.session_state["experiencia"],
                1,
                key='exp_input'
            )

        if st.session_state.experiencia == 0:
            st.session_state.siniestros = 0
            st.markdown("ℹ️ Sin experiencia → sin siniestros registrados.")
        else:
            st.session_state.siniestros = st.number_input(
                "Nº de siniestros (últimos 5 años)",
                0, 10,
                st.session_state["siniestros"],
                key='siniestros_input'
            )

    # --- Columna derecha: Información del Vehículo ---
    with col2:
        st.markdown("""
        <h6 style="
            display: flex; 
            align-items: center; 
            gap: 8px; 
            font-size: 1.1rem; 
            font-weight: 600; 
            color: #ffffff; 
            margin-bottom: 1rem;
        ">
            Información del Vehículo
        </h6>
        """, unsafe_allow_html=True)

        st.session_state.antiguedad = st.slider(
            "Antigüedad del coche",
            0, 30,
            st.session_state.get("antiguedad", 5),
            1,
            key='antiguedad_input'
        )
        antiguedad = st.session_state.antiguedad  # variable solo para legibilidad

        # --- Cálculo de máximo dinámico según antigüedad ---
        if antiguedad == 0:
            km_maximo = 50000
        else:
            km_maximo = min(400000, (antiguedad + 1) * 50000)

        # --- Slider de kilometraje respetando el máximo ---
        st.session_state.km = st.slider(
            f"Kilometraje total (máx: {km_maximo:,} km)",
            0, km_maximo,
            min(st.session_state.get("km", 50000), km_maximo),
            step=1000,
            key='km_input'
        )

    st.markdown("___")

    # --- Función para reiniciar ---
    def reiniciar_formulario():
        st.session_state.update({
            "edad": 35,
            "experiencia": 2,
            "km": 50000,
            "antiguedad": 5,
            "siniestros": 0,
            "prima_modelo": 0.0,
            "mostro_prima": False
        })
        safe_rerun()

    # --- Botones de acción ---
    if not st.session_state["mostro_prima"]:
        if st.button("🔮 Calcular Prima Estimada"):
            input_data = pd.DataFrame([{
                'edad': st.session_state.edad,
                'experiencia': st.session_state.experiencia,
                'kilometraje': st.session_state.km,
                'antiguedad_auto': st.session_state.antiguedad,
                'anio_fabricacion': 2025 - st.session_state.antiguedad,
                'siniestros_5y': st.session_state.siniestros
            }])
            prima_estim = xgb_model.predict(input_data)[0]
            st.session_state["prima_modelo"] = float(prima_estim)
            st.session_state["mostro_prima"] = True
            safe_rerun()
    else:
        st.success(f"💶 Prima estimada: **{st.session_state['prima_modelo']:.2f} €**")
        col_botones = st.columns([1, 1])
        with col_botones[0]:
            if st.button("🔄 Reiniciar"):
                reiniciar_formulario()
        with col_botones[1]:
            st.button("➡️ Siguiente", on_click=siguiente)



# ---------- Pestaña 2: Ajuste de riesgo ----------
elif st.session_state['pagina'] == 2:
    st.title("💰 Ajuste de Prima por Riesgo")
    st.markdown("Selecciona los factores relacionados con los hábitos de conducción. Estos se usarán para ajustar tu prima base según el nivel de riesgo por accidente asociado.")
    st.markdown("___")

    col1, col2 = st.columns(2)

    # --- Columna izquierda: Provincia y Zona ---
    with col1:
        provincia_nombre = st.selectbox("Selecciona la provincia de conducción más habitual", list(provincia_map.values()))
        provincia_codigo = [k for k, v in provincia_map.items() if v == provincia_nombre][0]

        zona_nombre = st.selectbox("Selecciona la zona predominante", list(zona_map.values()))
        zona_codigo = [k for k, v in zona_map.items() if v == zona_nombre][0]

    # --- Columna derecha: Franja y Tipo de día ---
    with col2:
        franjas = st.multiselect(
            "Selecciona los horarios de conducción más habituales",
            options=franja_riesgo['FRANJA_HORARIA'].unique(),
            default=[franja_riesgo['FRANJA_HORARIA'].unique()[0]]
        )

        dia_tipo = st.selectbox("Día más frecuente", diatipo_riesgo['DIA_TIPO'].unique())

    st.markdown("---")

    # --- Cálculo del ajuste ---
    if not st.session_state.get("ajuste_realizado", False):
        if st.button("⚖️ Calcular ajuste por riesgo"):
            # Multiplicadores individuales
            m1 = prov_riesgo[prov_riesgo['COD_PROVINCIA'] == provincia_codigo]['MULTIPLICADOR_RIESGO'].values[0]
            m2_list = franja_riesgo.loc[franja_riesgo['FRANJA_HORARIA'].isin(franjas),'MULTIPLICADOR_RIESGO'].values
            m2 = np.prod(m2_list) ** (1 / len(m2_list)) # Media geométrica de los multiplicadores de franjas
            m3 = zona_riesgo[zona_riesgo['ZONA_AGRUPADA'] == zona_codigo]['MULTIPLICADOR_RIESGO'].values[0]
            m4 = diatipo_riesgo[diatipo_riesgo['DIA_TIPO'] == dia_tipo]['MULTIPLICADOR_RIESGO'].values[0]

            pesos = [0.35, 0.30, 0.20, 0.15]
            multiplicador = round((m1**pesos[0]) * (m2**pesos[1]) * (m3**pesos[2]) * (m4**pesos[3]), 4)

            prima_base = st.session_state.get("prima_modelo", 0)
            prima_ajustada = round(prima_base * multiplicador, 2)

            # Guardar resultados en session_state
            st.session_state.update({
                "multiplicador": multiplicador,
                "prima_ajustada": prima_ajustada,
                "m1_provincia": m1,
                "m2_franja": m2,
                "m3_zona": m3,
                "m4_dia": m4,
                "provincia_nombre": provincia_nombre,
                "franjas": franjas,
                "zona_nombre": zona_nombre,
                "dia_tipo": dia_tipo,
                "ajuste_realizado": True
            })

            safe_rerun()

    else:
        # --- Mostrar resultados ---
        st.markdown("### Resultado Final")
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Multiplicador por hábitos", f"{st.session_state['multiplicador']}")
        col_res2.metric(
            "Prima ajustada",
            f"{st.session_state['prima_ajustada']:.2f} €",
            delta=f"{st.session_state['prima_ajustada'] - st.session_state['prima_modelo']:.2f} €"
        )

        # --- Botones de navegación ---
        col_nav = st.columns([1, 1])
        with col_nav[0]:
            st.button("⬅️ Atrás", on_click=anterior)
        with col_nav[1]:
            st.button("➡️ Siguiente", on_click=siguiente)


# --- SECCIÓN DE RESUMEN (Pestaña 3) ---

elif st.session_state['pagina'] == 3:

    # Función de reinicio completo
    def reiniciar_todo():
        st.session_state.update({
            "pagina": 1,
            "edad": 35,
            "experiencia": 2,
            "km": 50000,
            "antiguedad": 5,
            "siniestros": 0,
            "prima_modelo": 0.0,
            "multiplicador": 1.0,
            "prima_ajustada": 0.0,
            "mostro_prima": False,
            "ajuste_realizado": False
        })

    # Recuperar datos del session_state    
    edad = st.session_state.get("edad", 35)
    experiencia = st.session_state.get("experiencia", 2)
    siniestros = st.session_state.get("siniestros", 0)
    antiguedad_auto = st.session_state.get("antiguedad", 5)
    kilometraje = st.session_state.get("km", 50000)
    prima_base = st.session_state.get("prima_modelo", 0.0)
    multiplicador = st.session_state.get("multiplicador", 1.0)
    prima_ajustada = st.session_state.get("prima_ajustada", 0.0)
    provincia_nombre = st.session_state.get("provincia_nombre", "")
    franjas = st.session_state.get("franjas", [])
    zona_nombre = st.session_state.get("zona_nombre", "")
    dia_tipo = st.session_state.get("dia_tipo", "")
    m1 = st.session_state.get('m1_provincia', 1.0)
    m2 = st.session_state.get('m2_franja', 1.0)
    m3 = st.session_state.get('m3_zona', 1.0)
    m4 = st.session_state.get('m4_dia', 1.0)

    st.title(" 📋 Resumen de Resultados")
    
    st.markdown("___")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🧑 Perfil del Conductor**")
        st.markdown(f"- Edad: {edad} años")
        st.markdown(f"- Experiencia: {experiencia} años")
        st.markdown(f"- Siniestros (5 años): {siniestros}")
    with col2:
        st.markdown("**🚗 Datos del Vehículo**")
        st.markdown(f"- Antigüedad: {antiguedad_auto} años")
        st.markdown(f"- Kilometraje: {kilometraje:,} km")
        st.markdown(f"- Año fabricación: {2025 - antiguedad_auto}")

    st.markdown("___")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Prima Base", f"{prima_base:.2f} €")
    with col2:
        st.metric("Multiplicador", f"{multiplicador:.4f}")
    with col3:
        st.metric("Prima Final", f"{prima_ajustada:.2f} €", delta=f"{prima_ajustada - prima_base:.2f} €")

    st.markdown("___")

    st.markdown("### 🧠 Desglose de Factores de ajuste")
    st.markdown(f"""
    <style>
    .factor-mini {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        background: #1e1e1e;
        border-radius: 8px;
        padding: 10px;
        font-size: 0.9rem;
    }}
    .factor-mini div {{ padding: 4px 0; }}
    </style>

    <div class="factor-mini">
        <div>📍 Provincia:</div><div><b>{provincia_nombre}</b> ({m1:.4f})</div>
        <div>⏰ Franja Horaria:</div><div><b>{', '.join(franjas)}</b> ({m2:.4f})</div>
        <div>🏙️ Zona:</div><div><b>{zona_nombre}</b> ({m3:.4f})</div>
        <div>📅 Día:</div><div><b>{dia_tipo}</b> ({m4:.4f})</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("___")

    st.markdown("### 📊 Gráfico de radar")
    col_radar, col_desc = st.columns([1, 1.3])
    with col_radar:
        valores = {
            "Edad": (75 - edad) / (75 - 18),
            "Experiencia": experiencia / 30,
            "Antigüedad": antiguedad_auto / 30,
            "Kilometraje": kilometraje / 200000,
            "Siniestros": siniestros / 10
        }
        labels = list(valores.keys())
        stats = list(valores.values())
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        stats += stats[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'polar': True})
        ax.plot(angles, stats, color='#1f77b4', linewidth=2)
        ax.fill(angles, stats, color='#1f77b4', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels([])
        st.pyplot(fig)

    with col_desc:
        st.markdown("""
        <div style="background:#1e1e1e; padding:10px; border-radius:8px; font-size:0.85rem">
            <p>🔹 <b>Edad:</b> Más riesgo si < 25 años</p>
            <p>🔹 <b>Experiencia:</b> Menos riesgo con más años</p>
            <p>🔹 <b>Antigüedad:</b> >15 años = más averías</p>
            <p>🔹 <b>Kilometraje:</b> >150,000 km = más desgaste</p>
            <p>🔹 <b>Siniestros:</b> Aumentan tu prima</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("___")
    
    st.button("🔄 Reiniciar", on_click=reiniciar_todo, key="reiniciar3")
