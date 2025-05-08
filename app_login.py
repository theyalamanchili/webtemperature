import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO
from scipy.optimize import fsolve
import plotly.graph_objects as go

# -----------------------------------------------
# Background Styling
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

# -----------------------------------------------
# Persistent Footer
# -----------------------------------------------
def footer():
    st.markdown(
        """
        <div class='footer'>Version α1 | © 2025 Texas A&M University</div>
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
    half_t = t/2
    dT = T0 - Tinf
    beta = rho*c*v/(2*k)
    Bi   = h*half_t/k
    def char_eq(z): return np.tan(z) - Bi/z
    eps = 1e-6
    z = np.zeros(N)
    z[0] = fsolve(char_eq, [eps, np.pi/2-eps])[0]
    odds = np.arange(1, 2*N, 2)
    for i in range(1, N):
        lo = odds[i-1]*np.pi/2 + eps
        hi = lo + np.pi - 2*eps
        z[i] = fsolve(char_eq, [lo, hi])[0]
    lam = z/half_t
    a = np.array([(2*dT*np.sin(z[i]))/(z[i]+np.sin(z[i])*np.cos(z[i])) for i in range(N)])
    x = np.linspace(0, L, 600)
    y = np.linspace(-half_t, half_t, 300)
    X, Yg = np.meshgrid(x, y)
    Theta = sum(a[i]*np.exp((beta-np.sqrt(beta**2+lam[i]**2))*X)*np.cos(lam[i]*Yg) for i in range(N))
    return x, y, X, Yg, Tinf+Theta

def solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x):
    half_t = t/2
    dT = T0 - Tinf
    beta = rho*c*v/(2*k)
    A = 2*W*half_t
    P = 2*W + 2*half_t
    m2 = h*P/(k*A)
    mu = beta - np.sqrt(beta**2+m2)
    return Tinf + dT*np.exp(mu*x)

# -----------------------------------------------
# App Initialization
# -----------------------------------------------
add_styles()
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    login()
    footer()
    st.stop()
# Show readme by default
if 'show_readme' not in st.session_state:
    st.session_state.show_readme = True
if st.session_state.show_readme:
    exp = st.expander("📖 Read Me / User Guide", expanded=True)
    exp.markdown(
        """
**Description:**  
Analytical tool for steady-state temperature in a moving web under various transport scenarios using 2D and 1D models.

**Transport Scenario:**  
Defines how the web exchanges heat (e.g., free span convective cooling, roller contact, zone heating).

**Material Properties (vertical list):**  
- Thermal conductivity k [W/m·K]  
- Density ρ [kg/m³]  
- Specific heat c [J/kg·K]

**Web Transport Inputs:**  
- **Inlet Temperature T0 [°C]**: Upstream roller or inlet condition.
- **Ambient Temperature T∞ [°C]**: Surrounding fluid.
- **Convective HTC h [W/m²·K]**: Surface heat transfer.

**Transport & Process Parameters:**  
- **Speed v [m/s]**  
- **Thickness t [m]**  
- **Width W [m]**  
- **Span length L [m]**

**Default Parameters:**  
- **Number of eigenvalues N**: Series terms for 2D solution; increase for higher accuracy at cost of compute time.

**Outputs:**  
- **2D Contour** (X=span, Y=thickness)  
- **Profiles vs span** (avg, mid-plane, surfaces, 1D)  
- **Differences** (mid-top, avg-1D)  
- **Biot & Péclet numbers**  
- **CSV download**

**Disclaimer:**  
Assumes constant properties, steady-state, and ideal boundary conditions. Validate experimentally.

**Citation:**  
Yalamanchili, A. V.; Sagapuram, D.; Pagilla, P. R. (2024). "Modeling Steady-State Temperature Distribution in Moving Webs..." *Journal of Heat Transfer*. doi:10.1115/1.4051234
"""
    )
    if exp.button("Close Read Me"):
        st.session_state.show_readme = False
        st.experimental_rerun()
    footer()
    st.stop()

# Main UI
st.title("Web Temperature Distribution Simulator (α1)")

# Sidebar: Web Transport Scenario
st.sidebar.header("Web Transport Scenario")
scenario = st.sidebar.selectbox("Select scenario:",["Free span convective cooling","Web on heated/cooled roller","Web in heating/cooling zone"])
Y0 = st.sidebar.number_input("Inlet Temperature (T0) [°C]", -50.0,500.0,200.0)
Tinf = st.sidebar.number_input("Ambient Temperature (T∞) [°C]", -50.0,200.0,25.0)
h = st.sidebar.number_input("Convective HTC (h) [W/m²·K]", 1.0,10000.0,100.0)

# Sidebar: Material Properties
st.sidebar.header("Material Properties")
matlib = {'PET':(0.2,1390,1400),'Aluminum':(237,2700,897),'Copper':(401,8960,385),'Custom':None}
materials = list(matlib.keys())
mat = st.sidebar.selectbox("Material:",materials,index=0)
if mat!='Custom':
    k,rho,c = matlib[mat]
    st.sidebar.markdown(f"""
- k = {k} W/m·K  
- ρ = {rho} kg/m³  
- c = {c} J/kg·K
""")
else:
    k = st.sidebar.number_input("k [W/m·K]",0.1,500.0,0.2)
    rho = st.sidebar.number_input("ρ [kg/m³]",100,20000,1390)
    c = st.sidebar.number_input("c [J/kg·K]",100,5000,1400)

# Sidebar: Transport & Process Parameters
st.sidebar.header("Transport & Process Parameters")
v = st.sidebar.number_input("Speed v [m/s]",0.01,10.0,1.6)
t = st.sidebar.number_input("Thickness t [m]",1e-6,1e-2,0.001,step=1e-6,format="%.6f")
W = st.sidebar.number_input("Width W [m]",0.01,5.0,1.0)
L = st.sidebar.number_input("Span length L [m]",0.1,50.0,10.0)

# Sidebar: Default Parameters
st.sidebar.header("Default Parameters")
N = st.sidebar.slider("Number of eigenvalues N",5,50,20)
st.sidebar.markdown("Series terms for 2D model; increase for accuracy, at cost of compute.")

# Compute
if st.button("Compute"):
    if scenario=="Free span convective cooling":
        x,y,X,Yg,T2 = solve_2d(k,rho,c,v,Y0,Tinf,h,t,L,N)
        T1 = solve_1d(k,rho,c,v,Y0,Tinf,h,t,W,x)
        st.session_state.update(x=x,y=y,X=X,Yg=Yg,T2=T2,T1=T1)
        st.session_state.ready=True
    else:
        st.warning("Solver for this scenario is coming soon.")

# Results
if st.session_state.get('ready',False):
    x,y,X,Yg,T2,T1 = (st.session_state[var] for var in ['x','y','X','Yg','T2','T1'])
    Bi = h*(t/2)/k
    Pe = v*L/(k/(rho*c))
    st.subheader("2D Temperature Contour")
    st.markdown("**X:** span (m), **Y:** thickness (m)")
    if st.checkbox("Show contour lines"):
        contours=dict(showlines=True)
    else:
        contours=dict(showlines=False)
    fig=go.Figure(go.Contour(z=T2,x=x,y=y,ncontours=60,contours=contours))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown(f"**Biot:** {Bi:.2f}  
**Péclet:** {Pe:.1f}")

    st.subheader("Temperature Profiles vs Span")
    profiles={
        'Average':T2.mean(axis=0),
        'Mid-plane':T2[np.argmin(np.abs(y))],
        'Top surface':T2[np.argmin(np.abs(y-t/2))],
        'Bottom surface':T2[np.argmin(np.abs(y+t/2))],
        '1D model':T1
    }
    for label,data in profiles.items():
        if st.checkbox(label):
            fig2=go.Figure(go.Scatter(x=x,y=data,mode='lines+markers',name=label))
            st.plotly_chart(fig2,use_container_width=True)

    st.subheader("Temperature Differences vs Span")
    if st.checkbox("Mid-plane - Top surface"):
        d=profiles['Mid-plane']-profiles['Top surface']
        fig3=go.Figure(go.Scatter(x=x,y=d,mode='lines',name='Mid-Top'))
        st.plotly_chart(fig3,use_container_width=True)
    if st.checkbox("Avg - 1D model"):
        d=profiles['Average']-profiles['1D model']
        fig4=go.Figure(go.Scatter(x=x,y=d,mode='lines',name='Avg-1D'))
        st.plotly_chart(fig4,use_container_width=True)

    df=pd.DataFrame({'x':X.flatten(),'y':Yg.flatten(),'T':T2.flatten()})
    buf=BytesIO();df.to_csv(buf,index=False);buf.seek(0)
    st.download_button("Download CSV",buf,"temperature_data.csv","text/csv")

footer()
