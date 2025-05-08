import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO
from scipy.optimize import fsolve
import plotly.graph_objects as go

# -----------------------------------------------
# Background Styling and Footer
# -----------------------------------------------
def add_styles():
    st.markdown(
        """
        <style>
        .stApp { background-color: #f9f9f9; color: #111; }
        .footer { position: fixed; bottom: 0; width: 100%; text-align: center;
                  font-size: 0.8rem; color: #555; }
        </style>
        """, unsafe_allow_html=True)

def footer():
    st.markdown(
        """
        <div class='footer'>Version α-1 | © 2025 Texas A&amp;M University</div>
        """, unsafe_allow_html=True)

# -----------------------------------------------
# Login Screen
# -----------------------------------------------
def login():
    logo = Image.open("MEEN_logo.png")
    st.image(logo, width=300)
    st.header("Please log in to continue")

    user = st.text_input("Username")
    pwd  = st.text_input("Password", type="password")

    if st.button("Login"):
        creds = {"Guest":"GuestPass","Aditya":"Yalamanchili","Prabhakar":"Pagilla","admin":"adminpass"}
        if creds.get(user) == pwd:
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

# -----------------------------------------------
# 2D / 1D Solvers
# -----------------------------------------------

def solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N):
    half_thickness = t / 2
    dT = T0 - Tinf
    beta = rho * c * v / (2 * k)
    Bi   = h * half_thickness / k

    def fz(z_val): return np.tan(z_val) - Bi / z_val

    eps = 1e-6
    z = np.zeros(N)
    z[0] = fsolve(fz, [eps, np.pi/2 - eps])[0]
    odds = np.arange(1, 2*N, 2)
    for i in range(1, N):
        lo = odds[i-1] * np.pi/2 + eps
        hi = lo + np.pi - 2*eps
        z[i] = fsolve(fz, [lo, hi])[0]

    lam = z / half_thickness
    a   = np.array([(2 * dT * np.sin(z[i]))/(z[i] + np.sin(z[i])*np.cos(z[i])) for i in range(N)])

    x = np.linspace(0, L, 600)
    y = np.linspace(-half_thickness, half_thickness, 300)
    X, Yg = np.meshgrid(x, y)

    Theta = sum(a[i] * np.exp((beta - np.sqrt(beta**2 + lam[i]**2)) * X) * np.cos(lam[i] * Yg) for i in range(N))
    return x, y, X, Yg, Tinf + Theta


def solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x):
    half_thickness = t / 2
    dT = T0 - Tinf
    beta = rho * c * v / (2 * k)
    A = 2 * W * half_thickness
    P = 2 * W + 2 * half_thickness
    m2 = h * P / (k * A)
    mu = beta - np.sqrt(beta**2 + m2)
    return Tinf + dT * np.exp(mu * x)

# -----------------------------------------------
# Main Application
# -----------------------------------------------
add_styles()

# Init login state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    login()
    st.stop()

st.title("Web Temperature Distribution Simulator")

# Read-Me / User Guide
with st.expander("📖 Read Me / User Guide", expanded=False):
    st.markdown("""
**Description:**
Computes steady-state temperature distributions in a moving web using 2D and 1D analytical models.

**Usage:**
1. Select scenario, material, temperatures, and parameters in sidebar.  
2. Click **Compute** to display results and plots.  
3. Download CSV of temperature field.

**Model Assumptions:**
- Constant properties, steady-state, series truncation.  
- Validate experimentally for your setup.

**Citation:**  
Yalamanchili, A. V.; Sagapuram, D.; Pagilla, P. R. (2024). "Modeling Steady-State Temperature Distribution..."
""")

# Sidebar: Web Transport Scenario
st.sidebar.header("1. Web Transport Scenario")
scenario = st.sidebar.selectbox("Scenario:", [
    "Free span convective cooling",
    "Web on heated/cooled roller",
    "Web in heating/cooling zone"
])
if scenario == "Free span convective cooling":
    st.sidebar.image("BC2.png", use_container_width=True)
elif scenario == "Web on heated/cooled roller":
    st.sidebar.image("BC1.png", use_container_width=True)
else:
    st.sidebar.image("BC3.png", use_container_width=True)

# Sidebar: Temperatures & Convection
st.sidebar.header("2. Temperatures & Convection")
T0   = st.sidebar.number_input("Inlet temperature T₀ [°C]", -50.0, 500.0, 200.0)
Tinf = st.sidebar.number_input("Ambient temperature T∞ [°C]", -50.0, 200.0, 25.0)
h    = st.sidebar.number_input("Heat transfer coeff. h [W/(m²·K)]", 1.0, 10000.0, 100.0)

# Sidebar: Transport & Process Parameters
st.sidebar.header("3. Transport & Process Parameters")
v = st.sidebar.number_input("Web speed v [m/s]", 0.01, 10.0, 1.6)
t = st.sidebar.number_input("Thickness t [m]", 1e-6, 1e-2, 0.001)
W = st.sidebar.number_input("Width W [m]", 0.01, 5.0, 1.0)
L = st.sidebar.number_input("Span length L [m]", 0.1, 50.0, 10.0)

# Sidebar: Material Properties
st.sidebar.header("4. Material Properties")
matlib = {'PET':{'k':0.2,'rho':1390,'c':1400},'Aluminum':{'k':237,'rho':2700,'c':897},'Copper':{'k':401,'rho':8960,'c':385}}
materials = list(matlib.keys())+['Custom']
mat = st.sidebar.selectbox("Material:", materials, index=0)
if mat!='Custom':
    props=matlib[mat]
    st.sidebar.markdown(f"- k: **{props['k']}** W/(m·K)  \n- rho: **{props['rho']}** kg/m³  \n- c: **{props['c']}** J/(kg·K)")
    k,rho,c=props['k'],props['rho'],props['c']
else:
    k   = st.sidebar.number_input("k [W/(m·K)]",0.1,500.0,0.2)
    rho = st.sidebar.number_input("rho [kg/m³]",100,20000,1390)
    c   = st.sidebar.number_input("c [J/(kg·K)]",100,5000,1400)

# Sidebar: Default Parameters
st.sidebar.header("5. Default Parameters")
st.sidebar.markdown("# of eigenmodes for 2D solution; increase for accuracy vs compute time.")
N = st.sidebar.slider("Series terms N",5,50,20)

# Compute
if st.sidebar.button("Compute"):
    x,y,X,Yg,T2 = solve_2d(k,rho,c,v,T0,Tinf,h,t,L,N)
    T1 = solve_1d(k,rho,c,v,T0,Tinf,h,t,W,x)
    st.session_state.update(x=x,y=y,X=X,Yg=Yg,T2=T2,T1=T1)
    st.session_state.ready=True

# Display results
if st.session_state.get('ready', False):
    x,y,X,Yg,T2,T1 = (st.session_state[var] for var in ['x','y','X','Yg','T2','T1'])
    Yh=t/2; Bi_num=h*Yh/k; Pe_num=v*L/(k/(rho*c))

    st.subheader("2D Temperature Contour")
    st.markdown("**X-axis:** span position (m)  &nbsp; **Y-axis:** through-thickness (m)")
    show=st.checkbox("Show contour lines & labels")
    fig=go.Figure(go.Contour(z=T2,x=x,y=y,colorscale='Turbo',ncontours=60,contours=dict(showlines=show,showlabels=show)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Biot number:** {Bi_num:.2f}  &nbsp; **Péclet number:** {Pe_num:.1f}")

    st.subheader("Temperature Profiles vs Span")
    st.markdown("X-axis: span (m), Y-axis: temperature (°C)")
    Tavg=T2.mean(axis=0); Tmid=T2[np.argmin(np.abs(y))]; Ttop=T2[np.argmin(np.abs(y-Yh))]; Tbot=T2[np.argmin(np.abs(y+Yh))]
    profiles={'2D average':Tavg,'Mid-plane':Tmid,'Top surface':Ttop,'Bottom surface':Tbot,'1D model':T1}
    for label,data in profiles.items():
        if st.checkbox(label): st.plotly_chart(go.Figure(go.Scatter(x=x,y=data,mode='lines+markers',name=label)),use_container_width=True)

    st.subheader("Temperature Differences vs Span")
    if st.checkbox("Mid-plane minus Top surface"): st.plotly_chart(go.Figure(go.Scatter(x=x,y=profiles['Mid-plane']-profiles['Top surface'],mode='lines',name='Mid-Top')),use_container_width=True)
    if st.checkbox("2D average minus 1D model"): st.plotly_chart(go.Figure(go.Scatter(x=x,y=profiles['2D average']-profiles['1D model'],mode='lines',name='Avg-1D')),use_container_width=True)

    df=pd.DataFrame({'x':X.flatten(),'y':Yg.flatten(),'T':T2.flatten()});buf=BytesIO();df.to_csv(buf,index=False);buf.seek(0)
    st.download_button("Download CSV",buf,"temp_contour.csv","text/csv")

footer()
