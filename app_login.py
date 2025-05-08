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
    st.markdown(
        """
**Description:**  
Computes steady-state temperature distributions in a moving web using 2D and 1D analytical models.

**Usage:**  
1. Select inputs in sidebar.  
2. Click **Compute**.  
3. Download data/plots.  

**Assumptions:**  
- Constant properties, steady-state.  
- Series truncation; increase N for accuracy.

**Citation:**  
Yalamanchili et al. (2024). Modeling Steady-State Temperature Distribution...
"""
    )

# Sidebar Sections
st.sidebar.header("1. Web Transport Scenario")
scenario = st.sidebar.selectbox("Scenario:", [
    "Free span convective cooling",
    "Web on heated/cooled roller",
    "Web in heating/cooling zone"
])
if scenario == "Free span convective cooling": st.sidebar.image("BC2.png", use_container_width=True)
elif scenario == "Web on heated/cooled roller": st.sidebar.image("BC1.png", use_container_width=True)
else: st.sidebar.image("BC3.png", use_container_width=True)

st.sidebar.header("2. Material Properties")
matlib = {'PET':{'k':0.2,'rho':1390,'c':1400},'Aluminum':{'k':237,'rho':2700,'c':897},'Copper':{'k':401,'rho':8960,'c':385}}
materials = list(matlib.keys())+['Custom']
mat = st.sidebar.selectbox("Material:", materials, index=0)
if mat!='Custom':
    p = matlib[mat]
    st.sidebar.markdown(
        f"- k: **{p['k']}** W/(m·K)  \n"
        f"- rho: **{p['rho']}** kg/m³  \n"
        f"- c: **{p['c']}** J/(kg·K)"
    )
    k,rho,c = p['k'],p['rho'],p['c']
else:
    k   = st.sidebar.number_input("k [W/(m·K)]",0.1,500.0,0.2)
    rho = st.sidebar.number_input("rho [kg/m³]",100,20000,1390)
    c   = st.sidebar.number_input("c [J/(kg·K)]",100,5000,1400)

st.sidebar.header("3. Temperatures & Convection")
T0   = st.sidebar.number_input("Inlet T₀ [°C]", -50.0, 500.0, 200.0)
Tinf = st.sidebar.number_input("Ambient T∞ [°C]", -50.0, 200.0, 25.0)
h    = st.sidebar.number_input("h [W/(m²·K)]", 1.0, 10000.0, 100.0)

st.sidebar.header("4. Transport & Process Parameters")
v = st.sidebar.number_input("v [m/s]", 0.01, 10.0, 1.6)
t = st.sidebar.number_input("t [m]", 1e-6, 1e-2, 0.001)
W = st.sidebar.number_input("W [m]", 0.01, 5.0, 1.0)
L = st.sidebar.number_input("L [m]", 0.1, 50.0, 10.0)

st.sidebar.header("5. Default Parameters")
st.sidebar.markdown("# eigenmodes for 2D solution; increase for accuracy vs compute time.")
N = st.sidebar.slider("N",5,50,20)

# Compute
if st.sidebar.button("Compute"):
    x,y,X,Yg,T2 = solve_2d(k,rho,c,v,T0,Tinf,h,t,L,N)
    T1 = solve_1d(k,rho,c,v,T0,Tinf,h,t,W,x)
    st.session_state.update(x=x,y=y,X=X,Yg=Yg,T2=T2,T1=T1)
    st.session_state.ready=True

# Display results
if st.session_state.get('ready'):
    x,y,X,Yg,T2,T1 = (st.session_state[var] for var in ['x','y','X','Yg','T2','T1'])
    Yh=t/2; Bi=h*Yh/k; Pe=v*L/(k/(rho*c))

    st.subheader("2D Temperature Contour")
    st.markdown("**X-axis:** span (m) &nbsp; **Y-axis:** thickness (m)")
    show = st.checkbox("Show contour lines & labels")
    fig = go.Figure(go.Contour(z=T2,x=x,y=y,ncontours=60,contours=dict(showlines=show,showlabels=show)))
    st.plotly_chart(fig, use_container_width=True)

    # Download contour data
    df_contour = pd.DataFrame({'x':X.flatten(),'y':Yg.flatten(),'T':T2.flatten()})
    buf1=BytesIO();df_contour.to_csv(buf1,index=False);buf1.seek(0)
    st.download_button("Download Contour CSV",buf1,"contour.csv","text/csv")

    st.markdown(f"**Biot:** {Bi:.2f} &nbsp; **Péclet:** {Pe:.1f}")

    # Profile selection
    st.subheader("Temperature Profiles vs Span")
    sel_avg  = st.checkbox("2D average", value=True)
    sel_mid  = st.checkbox("Mid-plane", value=True)
    sel_top  = st.checkbox("Top surface", value=True)
    sel_bot  = st.checkbox("Bottom surface", value=True)
    sel_1d   = st.checkbox("1D model", value=True)

    fig_p = go.Figure()
    if sel_avg: fig_p.add_trace(go.Scatter(x=x,y=T2.mean(axis=0),mode='lines',name='2D avg'))
    if sel_mid: fig_p.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y))],mode='lines',name='Mid-plane'))
    if sel_top: fig_p.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y-Yh))],mode='lines',name='Top surface'))
    if sel_bot: fig_p.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y+Yh))],mode='lines',name='Bottom surface'))
    if sel_1d:  fig_p.add_trace(go.Scatter(x=x,y=T1,mode='lines',name='1D model',line=dict(dash='dash')))
    fig_p.update_layout(xaxis_title='Span (m)',yaxis_title='Temp (°C)',legend_title='Profiles')
    st.plotly_chart(fig_p, use_container_width=True)

    # Download profile data
    profile_data = pd.DataFrame({'x': x})
    if sel_avg: profile_data['avg'] = T2.mean(axis=0)
    if sel_mid: profile_data['mid'] = T2[np.argmin(np.abs(y))]
    if sel_top: profile_data['top'] = T2[np.argmin(np.abs(y-Yh))]
    if sel_bot: profile_data['bot'] = T2[np.argmin(np.abs(y+Yh))]
    if sel_1d:  profile_data['1d'] = T1
    buf2=BytesIO();profile_data.to_csv(buf2,index=False);buf2.seek(0)
    st.download_button("Download Profiles CSV",buf2,"profiles.csv","text/csv")

    # Differences selection
    st.subheader("Temperature Differences vs Span")
    sel_mt  = st.checkbox("Mid-plane – Top surface", value=True)
    sel_a1d = st.checkbox("2D avg – 1D model", value=True)
    fig_d = go.Figure()
    if sel_mt:  fig_d.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y))]-T2[np.argmin(np.abs(y-Yh))],mode='lines',name='Mid-Top'))
    if sel_a1d:fig_d.add_trace(go.Scatter(x=x,y=T2.mean(axis=0)-T1,mode='lines',name='Avg-1D'))
    fig_d.update_layout(xaxis_title='Span (m)',yaxis_title='ΔTemp (°C)',legend_title='Diffs')
    st.plotly_chart(fig_d, use_container_width=True)

    # Download differences data
    diff_data = pd.DataFrame({'x': x})
    if sel_mt:  diff_data['mid_top'] = T2[np.argmin(np.abs(y))] - T2[np.argmin(np.abs(y-Yh))]
    if sel_a1d:diff_data['avg_1d'] = T2.mean(axis=0) - T1
    buf3=BytesIO();diff_data.to_csv(buf3,index=False);buf3.seek(0)
    st.download_button("Download Differences CSV",buf3,"differences.csv","text/csv")

footer()
