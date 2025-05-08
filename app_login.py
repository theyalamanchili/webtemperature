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

# Initialize login state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    login()
    st.stop()

st.title("Web Temperature Distribution Simulator")

# Read-Me / User Guide
with st.expander("📖 Read Me / User Guide", expanded=False):
    st.markdown("""
**Overview & Background**  
In roll‑to‑roll (R2R) manufacturing, a continuous web of material travels through processing zones (e.g. ovens, cooling spans, rollers). Predicting its steady‑state temperature field is crucial for:
- Avoiding thermal defects (curl, wrinkles, web‑elongation)  
- Ensuring process uniformity  

This app solves two analytical models under *steady‑state* conditions:

1. **2D convection–diffusion** (across thickness _y_ and along span _x_):  
   
   $$ \frac{\partial^2 T}{\partial x^2} \;+\; \frac{\partial^2 T}{\partial y^2} \;-\; 2\,\beta\,\frac{\partial T}{\partial x} \;=\; 0, 
   \quad \beta=\frac{\rho\,c\,v}{2\,k}. $$

2. **1D lumped‑capacitance** (uniform through‑thickness approximation):  
   
   $$ \frac{d^2 T}{dx^2}
      \;-\;
      2\,\beta\,\frac{dT}{dx}
      \;-\;
      m^2\,(T - T_\infty)
      = 0,
   \quad m^2=\frac{h\,P}{k\,A}. $$

**Assumptions**  
- Material properties (_k_, _ρ_, _c_) constant  
- Steady‑state (no time dependence)  
- Idealized boundary conditions (convective cooling/roller contact)  
- Series truncation error ∝ 1/_N_ (increase _N_ for improved accuracy)

---

**How to Use**  
1. **Select Web Transport Scenario** (span cooling vs. roller contact vs. heating zone).  
2. **Set Material Properties** (choose library entry or custom _k_, _ρ_, _c_).  
3. **Enter Temperatures & Convection** (_T₀_, _T∞_, _h_).  
4. **Fill Transport & Process Parameters** (_v_, _t_, _W_, _L_).  
5. **Adjust Default Parameters** (number of eigenmodes _N_).  
6. Click **Compute**.  
7. Use the **checkboxes** under each plot to toggle curves.  
8. Download CSVs of the contour, profiles, or difference data.

**Computed Outputs**  

1. **2D Temperature Contour**  
   - X‑axis = span position (*x*) in m  
   - Y‑axis = through‑thickness (*y*) in m  
   - Color = local temperature *T(x,y)*  
   - Uses the “Turbo” color scale  

2. **Temperature Profiles vs. Span** (select via checkboxes):  
   - **Centerline:** \(T_c(x)=T(x,y=0)\)  
   - **Top surface:** \(T_{\rm top}(x)=T(x,y=+t/2)\)  
   - **Bottom surface:** \(T_{\rm bot}(x)=T(x,y=-t/2)\)  
   - **Thickness‑average:** \(\displaystyle T_{\rm avg}(x)=\frac{1}{t}\int_{-t/2}^{t/2}T(x,y)\,dy\)  
   - **1D Lumped Model:** \(T_{1D}(x)\)  

3. **Temperature Differences vs. Span** (select via checkboxes):  
   - \(\Delta T_{c-\rm top}(x)=T_c(x)-T_{\rm top}(x)\)  
   - \(\Delta T_{\rm avg-1D}(x)=T_{\rm avg}(x)-T_{1D}(x)\)  

4. **CSV Downloads**  
   - **Contour CSV:** full (x,y,T) field  
   - **Profiles CSV:** span vs. selected profiles  
   - **Differences CSV:** span vs. selected differences  

---

**Citation**  
Yalamanchili, A.V.; Pagilla, P.R. (2025). *Modeling Steady‑State Temperature Distribution in Moving Webs in Roll‑to‑Roll Manufacturing*.  
    """, unsafe_allow_html=False)


# Sidebar Sections
# 1. Web Transport Scenario
st.sidebar.header("1. Web Transport Scenario")
scenario = st.sidebar.selectbox(
    "Scenario:",
    ["Free span convective cooling", "Web on heated/cooled roller", "Web in heating/cooling zone"]
)
if scenario == "Free span convective cooling":
    st.sidebar.image("BC2.png", use_container_width=True)
elif scenario == "Web on heated/cooled roller":
    st.sidebar.image("BC1.png", use_container_width=True)
else:
    st.sidebar.image("BC3.png", use_container_width=True)

# 2. Material Properties
st.sidebar.header("2. Material Properties")
matlib = {'PET':{'k':0.2,'rho':1390,'c':1400}, 'Aluminum':{'k':237,'rho':2700,'c':897}, 'Copper':{'k':401,'rho':8960,'c':385}}
materials = list(matlib.keys()) + ['Custom']
mat = st.sidebar.selectbox("Material:", materials, index=0)
if mat != 'Custom':
    p = matlib[mat]
    st.sidebar.markdown(
        "Thermal conductivity, *k* (W·m⁻¹·K⁻¹): **%g**  \n"
        "Density, *ρ* (kg·m⁻³): **%g**  \n"
        "Specific heat, *c* (J·kg⁻¹·K⁻¹): **%g**" % (p['k'], p['rho'], p['c'])
    )
    k, rho, c = p['k'], p['rho'], p['c']
else:
    k   = st.sidebar.number_input("Thermal conductivity, *k* (W·m⁻¹·K⁻¹)", 0.1, 500.0, 0.2)
    rho = st.sidebar.number_input("Density, *ρ* (kg·m⁻³)", 100, 20000, 1390)
    c   = st.sidebar.number_input("Specific heat, *c* (J·kg⁻¹·K⁻¹)", 100, 5000, 1400)

# 3. Temperatures & Convection
st.sidebar.header("3. Temperatures & Convection")
T0   = st.sidebar.number_input("Inlet temperature, *T₀* (°C)", -50.0, 500.0, 200.0)
Tinf = st.sidebar.number_input("Ambient temperature, *T∞* (°C)", -50.0, 200.0, 25.0)
h    = st.sidebar.number_input("Convective coefficient, *h* (W·m⁻²·K⁻¹)", 1.0, 10000.0, 100.0)

# 4. Transport & Process Parameters
st.sidebar.header("4. Transport & Process Parameters")
v = st.sidebar.number_input("Web speed, *v* (m·s⁻¹)", 0.01, 10.0, 1.6)
t = st.sidebar.number_input("Thickness, *t* (m)", 1e-6, 1e-2, 0.001)
W = st.sidebar.number_input("Width, *W* (m)", 0.01, 5.0, 1.0)
L = st.sidebar.number_input("Span length, *L* (m)", 0.1, 50.0, 10.0)

# 5. Default Parameters (collapsible)
with st.sidebar.expander("5. Default Parameters", expanded=False):
    st.markdown("Number of eigenmodes for 2D solution; increase for accuracy vs compute time.")
    N = st.slider("Series terms, *N* (–)", 5, 50, 20)

# Compute button
if st.sidebar.button("Compute"):
    x, y, X, Yg, T2 = solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N)
    T1             = solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x)
    st.session_state.update(x=x, y=y, X=X, Yg=Yg, T2=T2, T1=T1)
    st.session_state.ready = True

# Display results
if st.session_state.get('ready'):
    x, y, X, Yg, T2, T1 = (st.session_state[var] for var in ['x','y','X','Yg','T2','T1'])
    Yh = t/2
    Bi = h * Yh / k
    Pe = v * L / (k / (rho * c))

    # Contour
    st.subheader("2D Temperature Contour")
    st.markdown("**X-axis:** span (m) &nbsp; **Y-axis:** thickness (m)")
    show = st.checkbox("Show contour lines & labels")
    fig = go.Figure(go.Contour(
        z=T2, x=x, y=y, colorscale='Turbo', ncontours=60,
        contours=dict(showlines=show, showlabels=show),
        colorbar=dict(title="Temperature (°C)")
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Download contour data
    df_cont = pd.DataFrame({'x':X.flatten(),'y':Yg.flatten(),'T':T2.flatten()})
    buf1 = BytesIO(); df_cont.to_csv(buf1,index=False); buf1.seek(0)
    st.download_button("Download Contour CSV", buf1, "contour.csv", "text/csv")

    st.markdown(f"**Biot number, *Bi* (–):** {Bi:.2f}  &nbsp; **Péclet number, *Pe* (–):** {Pe:.1f}")

    # Profile selection
    st.subheader("Temperature Profiles vs Span")
    selections = {
        '2D average':       st.checkbox("2D average", value=True),
        'Mid-plane':        st.checkbox("Mid-plane (y=0)", value=True),
        'Top surface':      st.checkbox("Top surface (y=+t/2)", value=True),
        'Bottom surface':   st.checkbox("Bottom surface (y=-t/2)", value=True),
        '1D model':         st.checkbox("1D model", value=True)
    }
    fig_p = go.Figure()
    if selections['2D average']:     fig_p.add_trace(go.Scatter(x=x,y=T2.mean(axis=0),mode='lines',name='2D avg'))
    if selections['Mid-plane']:      fig_p.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y))],mode='lines',name='Mid-plane'))
    if selections['Top surface']:    fig_p.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y-Yh))],mode='lines',name='Top surface'))
    if selections['Bottom surface']: fig_p.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y+Yh))],mode='lines',name='Bottom surface'))
    if selections['1D model']:       fig_p.add_trace(go.Scatter(x=x,y=T1,mode='lines',name='1D model',line=dict(dash='dash')))
    fig_p.update_layout(xaxis_title='Span (m)', yaxis_title='Temperature (°C)', legend_title='Profiles')
    st.plotly_chart(fig_p, use_container_width=True)

    # Download profile data
    df_prof = pd.DataFrame({'x': x})
    if selections['2D average']:     df_prof['2D_avg'] = T2.mean(axis=0)
    if selections['Mid-plane']:      df_prof['Mid_plane'] = T2[np.argmin(np.abs(y))]
    if selections['Top surface']:    df_prof['Top_surface'] = T2[np.argmin(np.abs(y-Yh))]
    if selections['Bottom surface']: df_prof['Bottom_surface'] = T2[np.argmin(np.abs(y+Yh))]
    if selections['1D model']:       df_prof['1D_model'] = T1
    buf2 = BytesIO(); df_prof.to_csv(buf2,index=False); buf2.seek(0)
    st.download_button("Download Profiles CSV", buf2, "profiles.csv", "text/csv")

    # Differences selection
    st.subheader("Temperature Differences vs Span")
    diff_sel = {
        'Mid-Top': st.checkbox("Mid-plane − Top surface", value=True),
        'Avg-1D':  st.checkbox("2D avg − 1D model", value=True)
    }
    fig_d = go.Figure()
    if diff_sel['Mid-Top']: fig_d.add_trace(go.Scatter(x=x,y=T2[np.argmin(np.abs(y))] - T2[np.argmin(np.abs(y-Yh))],mode='lines',name='Mid-Top'))
    if diff_sel['Avg-1D']:  fig_d.add_trace(go.Scatter(x=x,y=T2.mean(axis=0) - T1,mode='lines',name='Avg-1D'))
    fig_d.update_layout(xaxis_title='Span (m)', yaxis_title='ΔTemp (°C)', legend_title='Differences')
    st.plotly_chart(fig_d, use_container_width=True)

    # Download differences data
    df_diff = pd.DataFrame({'x': x})
    if diff_sel['Mid-Top']: df_diff['Mid_Top'] = T2[np.argmin(np.abs(y))] - T2[np.argmin(np.abs(y-Yh))]
    if diff_sel['Avg-1D']:  df_diff['Avg_1D']  = T2.mean(axis=0) - T1
    buf3 = BytesIO(); df_diff.to_csv(buf3,index=False); buf3.seek(0)
    st.download_button("Download Differences CSV", buf3, "differences.csv", "text/csv")

footer()
