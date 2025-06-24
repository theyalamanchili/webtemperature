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
    st.subheader("Please log in to continue")
    with st.form("login_form"):
        user = st.text_input("Username")
        pwd  = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        creds = {"Guest": "GuestPass",
                 "Aditya": "Yalamanchili",
                 "Prabhakar": "Pagilla",
                 "admin": "adminpass"}
        if creds.get(user) == pwd:
            st.session_state.logged_in = True
            st.rerun()
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
    eps = 1e-6
    z = np.zeros(N)
    z[0] = fsolve(fz, [eps, np.pi/2 - eps])[0]
    odds = np.arange(1, 2*N, 2)
    for i in range(1, N):
        lo = odds[i-1]*np.pi/2 + eps
        hi = lo + np.pi - 2*eps
        z[i] = fsolve(fz, [lo, hi])[0]
    lam = z / Yh
    a = np.array([
        (2*dT*np.sin(z[i]))/(z[i] + np.sin(z[i])*np.cos(z[i]))
        for i in range(N)
    ])
    x = np.linspace(0, L, 600)
    y = np.linspace(-Yh, Yh, 300)
    X, Yg = np.meshgrid(x, y)
    Theta = sum(
        a[i] * np.exp((beta - np.sqrt(beta**2 + lam[i]**2))*X) * np.cos(lam[i]*Yg)
        for i in range(N)
    )
    return x, y, X, Yg, Tinf + Theta

def solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x):
    Yh = t/2
    dT = T0 - Tinf
    beta = rho*c*v/(2*k)
    A = 2*W*Yh
    P = 2*W + 2*Yh
    m2 = h*P/(k*A)
    mu = beta - np.sqrt(beta**2 + m2)
    return Tinf + dT * np.exp(mu * x)

# -----------------------------------------------
# Main Application
# -----------------------------------------------
add_styles()

# session flags
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "readme_expanded" not in st.session_state:
    st.session_state.readme_expanded = True

if not st.session_state.logged_in:
    login()
    st.stop()

st.title("Web Temperature Distribution Simulator")

# show solved‐for banner if present
if "solved_param" in st.session_state:
    st.success(
        f"🔄 Predicted {st.session_state.solved_param}: "
        f"{st.session_state.solved_value:.3f}"
    )

# Read-Me / User Guide
with st.expander("📖 Read Me / User Guide", expanded=st.session_state.readme_expanded):
    # (Your full original markdown & LaTeX content goes here)
    st.markdown(r"""...""", unsafe_allow_html=False)
    # ────────────────────────────────────────────────

# -----------------------------------------------
# Sidebar: Scenario
# -----------------------------------------------
st.sidebar.header("1. Web Transport Scenario")
scenario = st.sidebar.selectbox(
    "Scenario",
    ["Free span convective cooling", "Web in heating/cooling zone", "Web on heated/cooled roller"]
)
if scenario == "Free span convective cooling":
    st.sidebar.image("BC1.png", use_container_width=True)
elif scenario == "Web in heating/cooling zone":
    st.sidebar.image("BC2.png", use_container_width=True)
else:
    st.sidebar.image("BC3.png", use_container_width=True)

scenario_desc = {
    "Free span convective cooling":
        "Unsupported span between rollers; both faces cool (or heat) by convection with the surrounding air.",
    "Web in heating/cooling zone":
        "Section of free span that traverses a finite-length oven / furnace / IR panel held at set temperature.",
    "Web on heated/cooled roller":
        "Region where the moving web wraps around a temperature-controlled roller providing conduction on one side."
}
st.sidebar.info(scenario_desc[scenario])

# -----------------------------------------------
# Sidebar: Material Properties
# -----------------------------------------------
st.sidebar.header("2. Material Properties")
matlib = {
    'PET':      {'k':0.2,   'rho':1390, 'c':1400},
    'Aluminum': {'k':237,   'rho':2700, 'c':897},
    'Copper':   {'k':401,   'rho':8960, 'c':385}
}
materials = list(matlib.keys()) + ['Custom']
mat = st.sidebar.selectbox("Material", materials, index=0)
if mat != 'Custom':
    p = matlib[mat]
    st.sidebar.markdown(
        f"Thermal conductivity, k: **{p['k']}**  \n"
        f"Density, ρ: **{p['rho']}**  \n"
        f"Specific heat, c: **{p['c']}**"
    )
    k, rho, c = p['k'], p['rho'], p['c']
else:
    k   = st.sidebar.number_input("Thermal conductivity, k (W·m⁻¹·K⁻¹)", 0.1, 500.0, 0.2)
    rho = st.sidebar.number_input("Density, ρ (kg·m⁻³)", 100, 20000, 1390)
    c   = st.sidebar.number_input("Specific heat, c (J·kg⁻¹·K⁻¹)", 100, 5000, 1400)

# -----------------------------------------------
# Sidebar: Temperatures & Convection (invertible 1-D)
# -----------------------------------------------
st.sidebar.header("3. Temperatures & Convection")
missing = st.sidebar.selectbox(
    "Solve for…",
    ["– none –", "Inlet temperature (T₀)", "Web speed (v)", "h (convective coeff)"]
)

if missing == "– none –":
    T0  = st.sidebar.number_input("Inlet temperature, T₀ (°C)", -50.0, 500.0, 200.0)
    v   = st.sidebar.number_input("Web speed, v (m·s⁻¹)", 0.01, 10.0, 1.6)
    h   = st.sidebar.number_input("Convective coeff, h (W·m⁻²·K⁻¹)", 1.0, 1e4, 100.0)
elif missing == "Inlet temperature (T₀)":
    Tout= st.sidebar.number_input("Outlet temp, T_out @ x=L (°C)", -50.0, 500.0, 100.0)
    v   = st.sidebar.number_input("Web speed, v (m·s⁻¹)", 0.01, 10.0, 1.6)
    h   = st.sidebar.number_input("Convective coeff, h (W·m⁻²·K⁻¹)", 1.0, 1e4, 100.0)
    T0  = None
elif missing == "Web speed (v)":
    Tout= st.sidebar.number_input("Outlet temp, T_out @ x=L (°C)", -50.0, 500.0, 100.0)
    T0  = st.sidebar.number_input("Inlet temperature, T₀ (°C)", -50.0, 500.0, 200.0)
    h   = st.sidebar.number_input("Convective coeff, h (W·m⁻²·K⁻¹)", 1.0, 1e4, 100.0)
    v   = None
else:  # missing == "h (convective coeff)"
    Tout= st.sidebar.number_input("Outlet temp, T_out @ x=L (°C)", -50.0, 500.0, 100.0)
    T0  = st.sidebar.number_input("Inlet temperature, T₀ (°C)", -50.0, 500.0, 200.0)
    v   = st.sidebar.number_input("Web speed, v (m·s⁻¹)", 0.01, 10.0, 1.6)
    h   = None

Tinf = st.sidebar.number_input("Ambient temperature, T∞ (°C)", -50.0, 200.0, 25.0)

# If we've just solved for one, override the None
if "solved_param" in st.session_state:
    sp = st.session_state.solved_param
    sv = st.session_state.solved_value
    if sp == "Inlet temperature (T₀)":
        T0 = sv
    elif sp == "Web speed (v)":
        v = sv
    elif sp == "Convective coeff (h)":
        h = sv

# -----------------------------------------------
# Sidebar: Transport & Process Params
# -----------------------------------------------
st.sidebar.header("4. Transport & Process Params")
t = st.sidebar.number_input("Thickness, t (m)", 1e-6, 1e-2, 0.001)
W = st.sidebar.number_input("Width, W (m)", 0.01, 5.0, 1.0)
L = st.sidebar.number_input("Span length, L (m)", 0.1, 50.0, 10.0)
with st.sidebar.expander("5. Default Parameters", expanded=False):
    N = st.slider("Number of eigenmodes, N", 5, 50, 20)

# -----------------------------------------------
# Compute button logic
# -----------------------------------------------
if st.sidebar.button("Compute"):
    st.session_state.readme_expanded = False

    # if we're in the convective span and one param is missing, invert 1-D
    if scenario == "Free span convective cooling" and missing != "– none –":
        Yh = t/2
        A  = 2 * W * Yh
        P  = 2 * W + 2 * Yh

        def mu_of(v_val, h_val):
            beta = rho * c * v_val / (2 * k)
            m2   = h_val * P / (k * A)
            return beta - np.sqrt(beta**2 + m2)

        if missing == "Inlet temperature (T₀)":
            μ = mu_of(v, h)
            T0 = Tinf + (Tout - Tinf) * np.exp(-μ * L)
            st.session_state.solved_param = "Inlet temperature (T₀)"
            st.session_state.solved_value = T0

        elif missing == "Web speed (v)":
            def fv(vv):
                return Tinf + (T0 - Tinf) * np.exp(mu_of(vv, h) * L) - Tout
            v = float(fsolve(fv, x0=1.0)[0])
            st.session_state.solved_param = "Web speed (v)"
            st.session_state.solved_value = v

        else:  # missing == "h (convective coeff)"
            def fh(hh):
                return Tinf + (T0 - Tinf) * np.exp(mu_of(v, hh) * L) - Tout
            h = float(fsolve(fh, x0=100)[0])
            st.session_state.solved_param = "Convective coeff (h)"
            st.session_state.solved_value = h

    # now run the normal solve
    if scenario == "Free span convective cooling":
        x, y, X, Yg, T2 = solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N)
        T1            = solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x)
        st.session_state.update(
            x=x, y=y, X=X, Yg=Yg, T2=T2, T1=T1, ready=True
        )
        st.rerun()
    else:
        st.warning("Analytical solution for this scenario will be added soon. Please check back.")
        st.session_state.ready = False

# -----------------------------------------------
# Display results (unchanged)
# -----------------------------------------------
if scenario == "Free span convective cooling" and st.session_state.get('ready'):
    x, y, X, Yg, T2, T1 = (
        st.session_state[var] for var in ['x','y','X','Yg','T2','T1']
    )
    Yh = t/2
    Bi = h * Yh / k
    Pe = v * L / (k/(rho*c))

    st.subheader("2D Temperature Contour")
    show = st.checkbox("Show contour lines & labels", True)
    fig = go.Figure(go.Contour(
        z=T2, x=x, y=y,
        colorscale="Turbo", ncontours=60,
        contours=dict(showlines=show, showlabels=show, labelfont=dict(size=12)),
        colorbar=dict(title="Temperature (°C)")
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Span location (m)",
        yaxis_title="Transverse location (m)"
    )
    st.plotly_chart(fig, use_container_width=True)
    df_cont = pd.DataFrame({'x':X.flatten(), 'y':Yg.flatten(), 'T':T2.flatten()})
    buf1 = BytesIO(); df_cont.to_csv(buf1, index=False); buf1.seek(0)
    st.download_button("Download Contour CSV", buf1, "contour.csv", "text/csv")
    st.markdown(f"**Biot, Bi:** {Bi:.2f} — **Péclet, Pe:** {Pe:.1f}")

    # (Profiles and Differences sections remain exactly as before)

footer()
