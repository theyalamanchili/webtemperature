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
# Login Screen
# -----------------------------------------------
def login():
    logo = Image.open("MEEN_logo.png")
    st.image(logo, width=300)
    st.header("Please log in to continue")

    user = st.text_input("Username")
    pwd  = st.text_input("Password", type="password")

    if st.button("Login"):
        creds = {
            "Guest":    "GuestPass",
            "Aditya":   "Yalamanchili",
            "Prabhakar":"Pagilla",
            "admin":    "adminpass"
        }
        if creds.get(user) == pwd:
            st.session_state.logged_in = True
            if hasattr(st, 'experimental_rerun'):
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
    odds = np.arange(1, 2 * N, 2)
    for i in range(1, N):
        lo = odds[i-1] * np.pi/2 + eps
        hi = lo + np.pi - 2 * eps
        z[i] = fsolve(fz, [lo, hi])[0]

    lam = z / half_thickness
    a   = np.array([
        (2 * dT * np.sin(z[i])) / (z[i] + np.sin(z[i]) * np.cos(z[i]))
        for i in range(N)
    ])

    x = np.linspace(0, L, 600)
    y = np.linspace(-half_thickness, half_thickness, 300)
    X, Yg = np.meshgrid(x, y)

    Theta = sum(
        a[i]
        * np.exp((beta - np.sqrt(beta**2 + lam[i]**2)) * X)
        * np.cos(lam[i] * Yg)
        for i in range(N)
    )
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

# Show login if needed
if not st.session_state.logged_in:
    login()
    st.stop()

# Title and Read Me button
col1, col2 = st.columns([8,1])
with col1:
    st.title("Web Temperature Distribution")
with col2:
    if st.button("Read Me"):
        st.session_state.show_modal = True

# Read Me modal
if st.session_state.get('show_modal', False):
    with st.modal("User Guide"):
        st.markdown("""
**Overview:**
This tool calculates steady-state temperature in a moving web under different cooling/heating setups using analytical 2D and 1D models.

**Inputs (all values should be realistic for your process):**
- **Case:** Select cooling/heating scenario.
- **Material:** Predefined (default PET) or enter custom k [W/(m·K)], rho [kg/m3], c [J/(kg·K)].
- **Web Speed:** v [m/s]
- **Inlet Temp:** T0 [degC]
- **Ambient Temp:** Tinf [degC]
- **Convective HTC:** h [W/(m2·K)]
- **Thickness:** t [m]
- **Width:** W [m]
- **Span Length:** L [m]
- **Series Terms:** N (eigenmodes for 2D solution)

**Outputs:**
- **2D Contour:** X-axis = span position (m), Y-axis = through-thickness (m).
- **Profiles:** Line plots vs span location:
  - *Average:* Mean through-thickness temperature from 2D.
  - *Mid-plane:* y=0 temperature.
  - *Top/Bottom:* y=±t/2 surface temperatures.
  - *1D model:* Lumped approximation.
- **Differences:**
  - *Mid-Top:* Mid-plane minus top surface.
  - *Avg-1D:* 2D average minus 1D solution.
- **Dimensionless Groups:** Biot = h·t/2/k, Péclet = v·L/(k/(rho·c)).
- **CSV download** of full temperature field.

**Notes & Disclaimer:**
- Assumes constant material properties, steady-state.
- Truncation error depends on N; increase for accuracy.
- Validate results experimentally.

[Close]
""")
        if st.button("Close Read Me"):
            st.session_state.show_modal = False

# Sidebar: Process setup
st.sidebar.header("Process Setup")
# Boundary case
case = st.sidebar.selectbox("Case:", [
    "Free span convective cooling",
    "Web on heated/cooled roller",
    "Web in heating/cooling zone"
])
# Material properties
st.sidebar.subheader("Material")
materials = ['PET', 'Aluminum', 'Copper', 'Custom']
mat = st.sidebar.selectbox("Select material:", materials, index=0)
if mat != 'Custom':
    mat_props = {'PET': (0.2,1390,1400), 'Aluminum':(237,2700,897), 'Copper':(401,8960,385)}
    k, rho, c = mat_props[mat]
    st.sidebar.write(f"k={k}, rho={rho}, c={c}")
else:
    k   = st.sidebar.number_input("Thermal conductivity k [W/(m·K)]", 0.1, 500.0, 0.2)
    rho = st.sidebar.number_input("Density rho [kg/m3]", 100, 20000, 1390)
    c   = st.sidebar.number_input("Specific heat c [J/(kg·K)]", 100, 5000, 1400)

# Process parameters
st.sidebar.subheader("Process Parameters")
v    = st.sidebar.number_input("Web speed v [m/s]", 0.01, 10.0, 1.6)
T0   = st.sidebar.number_input("Inlet temp T0 [degC]", -50.0, 500.0, 200.0)
Tinf = st.sidebar.number_input("Ambient temp Tinf [degC]", -50.0, 200.0, 25.0)
h    = st.sidebar.number_input("Convective HTC h [W/(m2·K)]", 1.0, 10000.0, 100.0)
t    = st.sidebar.number_input("Thickness t [m]", 1e-6, 1e-2, 0.001, step=1e-6, format="%.6f")
W    = st.sidebar.number_input("Width W [m]", 0.01, 5.0, 1.0)
L    = st.sidebar.number_input("Span length Let's L [m]", 0.1, 50.0, 10.0)
N    = st.sidebar.slider("Series terms N", 5, 50, 20)

# Compute button
if st.button("Compute"):  
    if case == "Free span convective cooling":
        x, y, X, Yg, T2 = solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N)
        T1            = solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x)
        st.session_state.update(x=x, y=y, X=X, Yg=Yg, T2=T2, T1=T1)
        st.session_state.ready = True
    else:
        st.warning("Solver for this case is coming soon.")

# Display results
if st.session_state.get('ready', False):
    x, y, X, Yg, T2, T1 = (st.session_state[var] for var in ['x','y','X','Yg','T2','T1'])

    # Dimensionless numbers
    Bi_num = h * (t/2) / k
    Pe_num = v * L / (k / (rho * c))

    st.subheader("2D Temperature Contour")
    st.markdown("X: span position (m), Y: through-thickness (m)")
    show = st.checkbox("Show contour lines")
    fig = go.Figure(go.Contour(z=T2, x=x, y=y, ncontours=60,
                               contours=dict(showlines=show)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"**Biot number:** {Bi_num:.2f}  \n"
        f"**Péclet number:** {Pe_num:.1f}"
    )

    st.subheader("Temperature Profiles vs Span")
    st.markdown("X-axis: span (m), Y-axis: temperature (degC)")
    options = {
        'Average through-thickness': T2.mean(axis=0),
        'Mid-plane (y=0)':            T2[np.argmin(np.abs(y))],
        'Top surface (y=+t/2)':       T2[np.argmin(np.abs(y - t/2))],
        'Bottom surface (y=-t/2)':    T2[np.argmin(np.abs(y + t/2))],
        '1D model':                   T1
    }
    for label, data in options.items():
        if st.checkbox(label):
            fig2 = go.Figure(go.Scatter(x=x, y=data, mode='lines+markers', name=label))
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Temperature Differences vs Span")
    st.markdown("Differences between selected profiles")
    if st.checkbox("Mid-plane minus Top surface"):
        delta = options['Mid-plane (y=0)'] - options['Top surface (y=+t/2)']
        fig3 = go.Figure(go.Scatter(x=x, y=delta, mode='lines', name='Mid-Top'))
        st.plotly_chart(fig3, use_container_width=True)
    if st.checkbox("Average minus 1D model"):
        delta = options['Average through-thickness'] - options['1D model']
        fig4 = go.Figure(go.Scatter(x=x, y=delta, mode='lines', name='Avg-1D'))
        st.plotly_chart(fig4, use_container_width=True)

    # Download data
    df = pd.DataFrame({'x': X.flatten(), 'y': Yg.flatten(), 'T': T2.flatten()})
    buf = BytesIO(); df.to_csv(buf, index=False); buf.seek(0)
    st.download_button("Download CSV", buf, "temperature_data.csv", "text/csv")

footer()
