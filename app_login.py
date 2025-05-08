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

    z = np.zeros(N)
    z[0] = fsolve(fz, [1e-6, np.pi/2 - 1e-6])[0]
    odds = np.arange(1, 2*N, 2)
    for i in range(1, N):
        lo = odds[i-1] * np.pi/2 + 1e-6
        hi = lo + np.pi - 2e-6
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

# Authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    login()
    st.stop()

st.title("Web Temperature Distribution Simulator")

# Read-Me
with st.expander("📖 Read Me / User Guide", expanded=False):
    st.markdown(
        """
**Description:** Computes steady-state web temperature in 2D/1D.  
**Assumptions:** Constant properties, steady-state, series truncation.  
**Citation:** Yalamanchili et al. (2024)
"""
    )

# Sidebar Inputs
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

st.sidebar.header("2. Material Properties (k, ρ, c)")
matlib = {'PET':{'k':0.2,'rho':1390,'c':1400},'Aluminum':{'k':237,'rho':2700,'c':897},'Copper':{'k':401,'rho':8960,'c':385}}
mats = list(matlib.keys())+['Custom']
mat = st.sidebar.selectbox("Material:", mats, index=0)
if mat!='Custom':
    p = matlib[mat]
    st.sidebar.write(f"k = {p['k']} W·m⁻¹·K⁻¹")
    st.sidebar.write(f"ρ = {p['rho']} kg·m⁻³")
    st.sidebar.write(f"c = {p['c']} J·kg⁻¹·K⁻¹")
    k,rho,c = p['k'],p['rho'],p['c']
else:
    k   = st.sidebar.number_input("k (thermal conductivity) [W·m⁻¹·K⁻¹]",0.1,500.0,0.2)
    rho = st.sidebar.number_input("ρ (density) [kg·m⁻³]",100,20000,1390)
    c   = st.sidebar.number_input("c (specific heat) [J·kg⁻¹·K⁻¹]",100,5000,1400)

st.sidebar.header("3. Temperatures & Convection")
T0   = st.sidebar.number_input("T₀ (inlet temp) [°C]",-50.0,500.0,200.0)
Tinf = st.sidebar.number_input("T∞ (ambient temp) [°C]",-50.0,200.0,25.0)
h    = st.sidebar.number_input("h (heat transfer) [W·m⁻²·K⁻¹]",1.0,10000.0,100.0)

st.sidebar.header("4. Transport & Process Params")
v = st.sidebar.number_input("v (web speed) [m·s⁻¹]",0.01,10.0,1.6)
t = st.sidebar.number_input("t (thickness) [m]",1e-6,1e-2,0.001)
W = st.sidebar.number_input("W (width) [m]",0.01,5.0,1.0)
L = st.sidebar.number_input("L (span length) [m]",0.1,50.0,10.0)

# Collapsible Default Params
with st.sidebar.expander("5. Default Parameters", expanded=False):
    st.write("N: # eigenmodes for 2D; increase for accuracy vs. compute time.")
    N = st.slider("N (series terms)",5,50,20)

# Compute
if st.sidebar.button("Compute"):
    x,y,X,Yg,T2 = solve_2d(k,rho,c,v,T0,Tinf,h,t,L,N)
    T1 = solve_1d(k,rho,c,v,T0,Tinf,h,t,W,x)
    st.session_state.update(x=x,y=y,X=X,Yg=Yg,T2=T2,T1=T1)
    st.session_state.ready=True

# Display
if st.session_state.get('ready'):
    x,y,X,Yg,T2,T1 = (st.session_state[var] for var in ['x','y','X','Yg','T2','T1'])
    Yh = t/2
    Bi = h*Yh/k
    Pe = v*L/(k/(rho*c))

    st.subheader("2D Temperature Contour")
    st.markdown("**X:** span (m) &nbsp; **Y:** thickness (m)")
    show = st.checkbox("Show contour lines & labels")
    fig = go.Figure(go.Contour(
        z=T2, x=x, y=y,
        colorscale='Turbo', colorbar=dict(title='T (°C)'),
        ncontours=60, contours=dict(showlines=show, showlabels=show)
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Contour data download
    dfc = pd.DataFrame({'x':X.flatten(),'y':Yg.flatten(),'T':T2.flatten()})
    b1=BytesIO();dfc.to_csv(b1,index=False);b1.seek(0)
    st.download_button("Download Contour CSV",b1,"contour.csv","text/csv")

    st.markdown(f"**Biot (Bi):** {Bi:.2f} &nbsp; **Péclet (Pe):** {Pe:.1f}")

    # Profiles
    st.subheader("Temperature Profiles vs Span")
    sel_avg = st.checkbox("2D avg", True)
    sel_mid = st.checkbox("Mid-plane", True)
    sel_top = st.checkbox("Top surface", True)
    sel_bot = st.checkbox("Bottom surface", True)
    sel_1d  = st.checkbox("1D model", True)
    figp = go.Figure()
    if sel_avg: figp.add_trace(go.Scatter(x=x,y=T2.mean(axis=0),mode='lines',name='2D avg'))
    if sel_mid: figp.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y))],mode='lines',name='Mid-plane'))
    if sel_top: figp.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y-Yh))],mode='lines',name='Top surface'))
    if sel_bot: figp.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y+Yh))],mode='lines',name='Bottom surface'))
    if sel_1d:  figp.add_trace(go.Scatter(x=x,y=T1,mode='lines',name='1D model',line=dict(dash='dash')))
    figp.update_layout(xaxis_title='Span (m)',yaxis_title='Temp (°C)',legend_title='Profiles')
    st.plotly_chart(figp, use_container_width=True)

    # Profiles data download
    dp = pd.DataFrame({'x': x})
    if sel_avg: dp['avg'] = T2.mean(axis=0)
    if sel_mid: dp['mid'] = T2[np.argmin(np.abs(y))]
    if sel_top: dp['top'] = T2[np.argmin(np.abs(y-Yh))]
    if sel_bot: dp['bot'] = T2[np.argmin(np.abs(y+Yh))]
    if sel_1d:  dp['one_d'] = T1
    b2=BytesIO();dp.to_csv(b2,index=False);b2.seek(0)
    st.download_button("Download Profiles CSV",b2,"profiles.csv","text/csv")

    # Differences
    st.subheader("Temperature Differences vs Span")
    sel_mt = st.checkbox("Mid-plane − Top surface", True)
    sel_a1d= st.checkbox("2D avg − 1D model", True)
    fd = go.Figure()
    if sel_mt:  fd.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y))]-T2[np.argmin(np.abs(y-Yh))],mode='lines',name='Mid-Top'))
    if sel_a1d:fd.add_trace(go.Scatter(x=x,y=T2.mean(axis=0)-T1,mode='lines',name='Avg-1D'))
    fd.update_layout(xaxis_title='Span (m)',yaxis_title='ΔT (°C)',legend_title='Differences')
    st.plotly_chart(fd, use_container_width=True)

    # Differences data download
    dd = pd.DataFrame({'x': x})
    if sel_mt:  dd['mid_top'] = T2[np.argmin(np.abs(y))] - T2[np.argmin(np.abs(y-Yh))]
    if sel_a1d:dd['avg_1d']  = T2.mean(axis=0) - T1
    b3=BytesIO();dd.to_csv(b3,index=False);b3.seek(0)
    st.download_button("Download Differences CSV",b3,"differences.csv","text/csv")

footer()
