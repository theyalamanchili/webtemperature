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
            if hasattr(st, 'experimental_rerun'):
                st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

# -----------------------------------------------
# Solvers
# -----------------------------------------------
def solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N):
    Yh = t/2
    dT = T0 - Tinf
    beta = rho*c*v/(2*k)
    Bi = h*Yh/k
    def fz(z): return np.tan(z) - Bi/z
    eps=1e-6
    z=np.zeros(N)
    z[0]=fsolve(fz,[eps,np.pi/2-eps])[0]
    odds=np.arange(1,2*N,2)
    for i in range(1,N):
        lo=odds[i-1]*np.pi/2+eps
        hi=lo+np.pi-2*eps
        z[i]=fsolve(fz,[lo,hi])[0]
    lam=z/Yh
    a=np.array([(2*dT*np.sin(z[i]))/(z[i]+np.sin(z[i])*np.cos(z[i])) for i in range(N)])
    x=np.linspace(0,L,600)
    y=np.linspace(-Yh,Yh,300)
    X,Yg=np.meshgrid(x,y)
    Theta=sum(a[i]*np.exp((beta-np.sqrt(beta**2+lam[i]**2))*X)*np.cos(lam[i]*Yg) for i in range(N))
    return x,y,X,Yg, Tinf+Theta

def solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x):
    Yh=t/2; dT=T0-Tinf; beta=rho*c*v/(2*k)
    A=2*W*Yh; P=2*W+2*Yh; m2=h*P/(k*A)
    mu=beta-np.sqrt(beta**2+m2)
    return Tinf + dT*np.exp(mu*x)

# -----------------------------------------------
# Main Application
# -----------------------------------------------
add_styles()
if 'logged_in' not in st.session_state:
    st.session_state.logged_in=False
if not st.session_state.logged_in:
    login()
    st.stop()
st.title("Web Temperature Distribution Simulator")

# Read-Me / User Guide
with st.expander("📖 Read Me / User Guide", expanded=False):
    st.markdown("""
**Overview & Background**  
In roll-to-roll manufacturing, a moving web reaches a steady-state temperature field where convective cooling along the span balances conduction through thickness.  

**2D convection–diffusion model** (across _y_ and along _x_):
""", unsafe_allow_html=False)
    st.latex(r"""
\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} - 2\beta \frac{\partial T}{\partial x} = 0\\
\beta=\frac{\rho \, c \, v}{2 \, k}
"""
    )
    st.markdown("""
**1D lumped‑capacitance model** (uniform through‑thickness):
""", unsafe_allow_html=False)
    st.latex(r"""
\frac{d^2 T}{dx^2} - 2\beta \frac{d T}{dx} - m^2\bigl(T - T_\infty\bigr) = 0\\
 m^2 = \frac{h \, P}{k \, A}
"""
    )
    st.markdown("""
**Assumptions**: constant _k_,_ρ_,_c_; steady-state (no time term); ideal boundaries; series truncation error ∝1/_N_.

---

**How to Use**  
1. Complete **sidebar** sections 1–5.  
2. Click **Compute**.  
3. Toggle curves via checkboxes under each plot.  
4. Download CSV data for contours, profiles, and differences.

**Input Variables**: as labeled in sidebar (with notation in italics and units).

**Computed Outputs**:
- **2D Temperature Contour** with Turbo colorbar and labels.  
- **Temperature Profiles** along span at Centerline, Top/Bottom surfaces, Thickness-average, and 1D Lumped.  
- **Temperature Differences**: Centerline−Top, Average−1D.  
- **Dimensionless numbers** Bi and Pe shown below contour.
---
**Citation**: Yalamanchili, A.V.; Pagilla, P.R. (2025). *Modeling Steady‑State Temperature Distribution in Moving Webs in Roll‑to‑Roll Manufacturing*.
    """, unsafe_allow_html=False)

# Sidebar Sections
st.sidebar.header("1. Web Transport Scenario")
scenario=st.sidebar.selectbox("Scenario",["Free span convective cooling","Web on heated/cooled roller","Web in heating/cooling zone"])
if scenario=="Free span convective cooling": st.sidebar.image("BC2.png",use_container_width=True)
elif scenario=="Web on heated/cooled roller": st.sidebar.image("BC1.png",use_container_width=True)
else: st.sidebar.image("BC3.png",use_container_width=True)

st.sidebar.header("2. Material Properties")
matlib={'PET':{'k':0.2,'rho':1390,'c':1400},'Aluminum':{'k':237,'rho':2700,'c':897},'Copper':{'k':401,'rho':8960,'c':385}}
materials=list(matlib.keys())+['Custom']
mat=st.sidebar.selectbox("Material",materials,index=0)
if mat!='Custom':
    p=matlib[mat]
    st.sidebar.markdown(
        f"Thermal conductivity, *k* (W·m⁻¹·K⁻¹): **{p['k']}**  \n"
        f"Density, *ρ* (kg·m⁻³): **{p['rho']}**  \n"
        f"Specific heat, *c* (J·kg⁻¹·K⁻¹): **{p['c']}**"
    )
    k,rho,c=p['k'],p['rho'],p['c']
else:
    k=st.sidebar.number_input("Thermal conductivity, *k* (W·m⁻¹·K⁻¹)",0.1,500.0,0.2)
    rho=st.sidebar.number_input("Density, *ρ* (kg·m⁻³)",100,20000,1390)
    c=st.sidebar.number_input("Specific heat, *c* (J·kg⁻¹·K⁻¹)",100,5000,1400)

st.sidebar.header("3. Temperatures & Convection")
T0=st.sidebar.number_input("Inlet temperature, *T₀* (°C)",-50.0,500.0,200.0)
Tinf=st.sidebar.number_input("Ambient temperature, *T∞* (°C)",-50.0,200.0,25.0)
h=st.sidebar.number_input("Convective coef, *h* (W·m⁻²·K⁻¹)",1.0,1e4,100.0)

st.sidebar.header("4. Transport & Process Params")
v=st.sidebar.number_input("Web speed, *v* (m·s⁻¹)",0.01,10.0,1.6)
t=st.sidebar.number_input("Thickness, *t* (m)",1e-6,1e-2,0.001)
W=st.sidebar.number_input("Width, *W* (m)",0.01,5.0,1.0)
L=st.sidebar.number_input("Span length, *L* (m)",0.1,50.0,10.0)

with st.sidebar.expander("5. Default Params",expanded=False):
    st.markdown("Number of eigenmodes, *N* (–): increase for accuracy vs. compute time.")
    N=st.slider("Series terms, *N* (–)",5,50,20)

# Compute
if st.sidebar.button("Compute"):
    x,y,X,Yg,T2=solve_2d(k,rho,c,v,T0,Tinf,h,t,L,N)
    T1=solve_1d(k,rho,c,v,T0,Tinf,h,t,W,x)
    st.session_state.update(x=x,y=y,X=X,Yg=Yg,T2=T2,T1=T1)
    st.session_state.ready=True

# Display
if st.session_state.get('ready'):
    x,y,X,Yg,T2,T1=(st.session_state[var] for var in ['x','y','X','Yg','T2','T1'])
    Yh=t/2; Bi=h*Yh/k; Pe=v*L/(k/(rho*c))

    st.subheader("2D Temperature Contour")
    st.markdown("**X-axis:** span (m) — **Y-axis:** thickness (m)")
    show=st.checkbox("Show contour labels",value=True)
    fig=go.Figure(go.Contour(
        z=T2,x=x,y=y,colorscale='Turbo',ncontours=60,
        contours=dict(showlines=True,showlabels=show,labelfont=dict(size=12)),
        colorbar=dict(title="Temperature (°C)")
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,use_container_width=True)
    df_cont=pd.DataFrame({'x':X.flatten(),'y':Yg.flatten(),'T':T2.flatten()})
    buf1=BytesIO();df_cont.to_csv(buf1,index=False);buf1.seek(0)
    st.download_button("Download Contour CSV",buf1,"contour.csv","text/csv")
    st.markdown(f"**Biot, *Bi*:** {Bi:.2f} — **Péclet, *Pe*:** {Pe:.1f}")

    # Profiles
    st.subheader("Temperature Profiles vs Span")
    st.markdown("Curves show *T(x)* at key thickness locations along the span.")
    sel={ 'Centerline':st.checkbox('Centerline (y=0)',True), 'Top surface':st.checkbox('Top surface (y=+t/2)',True), 'Bottom surface':st.checkbox('Bottom surface (y=-t/2)',True), 'Average':st.checkbox('Thickness-average',True), '1D Lumped':st.checkbox('1D Lumped Model',True) }
    figp=go.Figure()
    if sel['Centerline']: figp.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y))],mode='lines',name='Centerline'))
    if sel['Top surface']: figp.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y-Yh))],mode='lines',name='Top surface'))
    if sel['Bottom surface']: figp.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y+Yh))],mode='lines',name='Bottom surface'))
    if sel['Average']: figp.add_trace(go.Scatter(x=x,y=T2.mean(axis=0),mode='lines',name='Thickness-average'))
    if sel['1D Lumped']: figp.add_trace(go.Scatter(x=x,y=T1,mode='lines',name='1D Lumped',line=dict(dash='dash')))
    figp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Span (m)', yaxis_title='Temperature (°C)', legend_title='Profiles')
    st.plotly_chart(figp,use_container_width=True)
    df_prof=pd.DataFrame({'x':x})
    if sel['Centerline']: df_prof['centerline']=T2[np.argmin(np.abs(y))]
    if sel['Top surface']: df_prof['top_surface']=T2[np.argmin(np.abs(y-Yh))]
    if sel['Bottom surface']: df_prof['bottom_surface']=T2[np.argmin(np.abs(y+Yh))]
    if sel['Average']: df_prof['avg']=T2.mean(axis=0)
    if sel['1D Lumped']: df_prof['lumped']=T1
    buf2=BytesIO();df_prof.to_csv(buf2,index=False);buf2.seek(0)
    st.download_button("Download Profiles CSV",buf2,"profiles.csv","text/csv")

    # Differences
    st.subheader("Temperature Differences vs Span")
    st.markdown("Difference curves to highlight deviations between profiles.")
    ds={ 'Centerline-Top':st.checkbox('Centerline – Top surface',True), 'Avg-1D':st.checkbox('Thickness-average – 1D Lumped',True) }
    figd=go.Figure()
    if ds['Centerline-Top']: figd.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y))]-T2[np.argmin(np.abs(y-Yh))],mode='lines',name='Centerline–Top'))
    if ds['Avg-1D']: figd.add_trace(go.Scatter(x=x,y=T2.mean(axis=0)-T1,mode='lines',name='Avg–1D'))
    figd.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Span (m)', yaxis_title='ΔTemperature (°C)', legend_title='Differences')
    st.plotly_chart(figd,use_container_width=True)
    df_diff=pd.DataFrame({'x':x})
    if ds['Centerline-Top']: df_diff['ctop']=T2[np.argmin(np.abs(y))]-T2[np.argmin(np.abs(y-Yh))]
    if ds['Avg-1D']: df_diff['avg1d']=T2.mean(axis=0)-T1
    buf3=BytesIO();df_diff.to_csv(buf3,index=False);buf3.seek(0)
    st.download_button("Download Differences CSV",buf3,"differences.csv","text/csv")

footer()
