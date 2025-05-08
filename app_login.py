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
            "Guest": "GuestPass",
            "Aditya": "Yalamanchili",
            "Prabhakar": "Pagilla",
            "admin": "adminpass"
        }
        if creds.get(user) == pwd:
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password.")

# -----------------------------------------------
# Persistent Footer
# -----------------------------------------------
def footer():
    st.markdown(
        """
        <div class='footer'>Version α 0.1 | © 2025 Texas A&amp;M University</div>
        """, unsafe_allow_html=True)

# -----------------------------------------------
# Solver: 2D convective free span
# -----------------------------------------------
def solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N):
    Y = t/2
    dT = T0 - Tinf
    beta = rho * c * v / (2 * k)
    Bi   = h * Y / k

    def fz(z): return np.tan(z) - Bi / z

    eps = 1e-6
    z = np.zeros(N)
    z[0] = fsolve(fz, [eps, np.pi/2 - eps])[0]
    odds = np.arange(1, 2*N, 2)
    for i in range(1, N):
        lo = odds[i-1] * np.pi/2 + eps
        hi = lo + np.pi - 2*eps
        z[i] = fsolve(fz, [lo, hi])[0]

    lam = z / Y
    a   = np.array([
        (2*dT * np.sin(z[i])) /
        (z[i] + np.sin(z[i]) * np.cos(z[i]))
        for i in range(N)
    ])

    # finer grid for smooth contours
    x = np.linspace(0, L, 600)
    y = np.linspace(-Y, Y, 300)
    X, Yg = np.meshgrid(x, y)

    Theta = sum(
        a[i]
        * np.exp((beta - np.sqrt(beta**2 + lam[i]**2)) * X)
        * np.cos(lam[i] * Yg)
        for i in range(N)
    )
    return x, y, X, Yg, (Tinf + Theta)

# -----------------------------------------------
# Solver: 1D analytical
# -----------------------------------------------
def solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x):
    Y = t/2
    dT = T0 - Tinf
    beta = rho * c * v / (2 * k)
    A = 2 * W * Y
    P = 2 * W + 2 * Y
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

# Show login until successful
if not st.session_state.logged_in:
    login()
    st.stop()

# Main UI after login
st.title("Web Temperature Distribution Simulator")

# ---- Boundary Condition ----
st.sidebar.header("Boundary Condition")
bc = st.sidebar.selectbox(
    "Select boundary condition:",
    [
        "Free span convective cooling",
        "Web over heated/cooled roller",
        "Web in heating/cooling zone"
    ]
)

if bc == "Free span convective cooling":
    st.image("BC2.png", caption="Free span convective cooling", use_column_width=True)
    st.sidebar.markdown(
        "_Free span with convective cooling on both surfaces. Enter thickness, speed, and h._"
    )
elif bc == "Web over heated/cooled roller":
    st.image("BC1.png", caption="Web over heated/cooled roller", use_column_width=True)
    st.sidebar.markdown(
        "_Web contacting a roller at fixed temperature. Inputs: roller T and contact length._"
    )
else:
    st.image("BC3.png", caption="Web in heating/cooling zone", use_column_width=True)
    st.sidebar.markdown(
        "_Web traveling through a heating/cooling zone. Inputs: zone T and length._"
    )

# ---- Material Properties ----
st.sidebar.header("Material Properties")
matlib = {
    'Aluminum': {'k': 237, 'rho': 2700, 'c': 897},
    'Copper':   {'k': 401, 'rho': 8960, 'c': 385},
    'PET':      {'k': 0.2, 'rho': 1390, 'c': 1400}
}
mat = st.sidebar.selectbox(
    "Material", list(matlib.keys()) + ['Custom']
)
if mat != 'Custom':
    k, rho, c = matlib[mat].values()
    st.sidebar.write(f"k={k} W/m·K, ρ={rho} kg/m³, c={c} J/kg·K")
else:
    k   = st.sidebar.number_input("Thermal conductivity k [W/m·K]", 0.1, 500.0, 0.2)
    rho = st.sidebar.number_input("Density ρ [kg/m³]", 100, 20000, 1400)
    c   = st.sidebar.number_input("Specific heat c [J/kg·K]", 100, 5000, 1400)

st.sidebar.markdown("---")

# ---- Process Parameters ----
st.sidebar.header("Process Parameters")
v    = st.sidebar.number_input("Web speed v [m/s]", 0.01, 10.0, 1.6)
T0   = st.sidebar.number_input("Inlet temp T₀ [°C]", -50.0, 500.0, 200.0)
Tinf = st.sidebar.number_input("Ambient temp T∞ [°C]", -50.0, 200.0, 25.0)
h    = st.sidebar.number_input("Convective coeff h [W/m²·K]", 1.0, 10000.0, 100.0)
t    = st.sidebar.number_input(
    "Web thickness t [m]", 1e-6, 1e-2, 0.001,
    step=1e-6, format="%.6f"
)
W    = st.sidebar.number_input("Web width W [m]", 0.01, 5.0, 1.0)
L    = st.sidebar.number_input("Span length L [m]", 0.1, 50.0, 10.0)
N    = st.sidebar.slider("Series terms N", 5, 50, 20)

# ---- Compute button ----
if st.button("Compute"):
    if bc == "Free span convective cooling":
        x, y, X, Yg, T2 = solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N)
        T1 = solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x)
        st.session_state.update(x=x, y=y, X=X, Yg=Yg, T2=T2, T1=T1)
        st.session_state.ready = True
    else:
        st.warning("Solver for the selected boundary condition is coming soon.")

# ---- Display results ----
if st.session_state.get('ready', False):
    x, y, X, Yg, T2, T1 = (
        st.session_state[key]
        for key in ['x','y','X','Yg','T2','T1']
    )

    # Dimensionless numbers
    Yh = t/2
    Bi_num = h * Yh / k
    Pe_num = v * L / (k / (rho * c))

    # 2D Contour
    st.subheader("2D Temperature Contour")
    show_lines = st.checkbox("Show contour lines and labels")
    fig = go.Figure(
        go.Contour(
            z=T2, x=x, y=y,
            colorscale='Turbo', ncontours=60,
            contours=dict(
                showlines=show_lines,
                showlabels=show_lines,
                labelfont=dict(size=12, color='black')
            )
        )
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # Biot & Peclet
    st.markdown(f"""**Biot number (Bi):** {Bi_num:.2f}  
**Péclet number (Pe):** {Pe_num:.1f}"""):** {Pe_num:.1f}")
    st.markdown(
        """
        Biot indicates surface vs conduction; low Bi means conduction-dominated cooling.  
        Péclet indicates advection vs conduction; high Pe means advection-dominated cooling.
        """
    )

    # Temperature Profiles
    st.subheader("Temperature Profiles")
    idx_mid = np.argmin(np.abs(y))
    idx_top = np.argmin(np.abs(y - Yh))
    idx_bot = np.argmin(np.abs(y + Yh))
    Tavg = T2.mean(axis=0)
    Tmid = T2[idx_mid]
    Ttop = T2[idx_top]
    Tbot = T2[idx_bot]

    # Styling for lines
    line_styles = {
        '2D avg': {'color':'blue',   'dash':'solid'},
        'Mid-plane':{'color':'green', 'dash':'dash'},
        'Top surface':{'color':'red',    'dash':'dot'},
        'Bot surface':{'color':'purple', 'dash':'dashdot'},
        '1D soln':{'color':'black', 'dash':'longdash'}
    }
    mark_syms = {
        '2D avg':'circle', 'Mid-plane':'square',
        'Top surface':'triangle-up','Bot surface':'triangle-down','1D soln':'x'
    }

    show_avg = st.checkbox("Show 2D average")
    show_mid = st.checkbox("Show mid-plane")
    show_top = st.checkbox("Show top surface")
    show_bot = st.checkbox("Show bot surface")
    show_1d = st.checkbox("Show 1D soln")

    if any([show_avg, show_mid, show_top, show_bot, show_1d]):
        fig2 = go.Figure()
        if show_avg:
            fig2.add_trace(go.Scatter(
                x=x, y=Tavg, mode='lines+markers', name='2D avg',
                line=line_styles['2D avg'],
                marker=dict(symbol=mark_syms['2D avg'], size=6, color=line_styles['2D avg']['color'])
            ))
        if show_mid:
            fig2.add_trace(go.Scatter(
                x=x, y=Tmid, mode='lines+markers', name='Mid-plane',
                line=line_styles['Mid-plane'],
                marker=dict(symbol=mark_syms['Mid-plane'], size=6, color=line_styles['Mid-plane']['color'])
            ))
        if show_top:
            fig2.add_trace(go.Scatter(
                x=x, y=Ttop, mode='lines+markers', name='Top surface',
                line=line_styles['Top surface'],
                marker=dict(symbol=mark_syms['Top surface'], size=6, color=line_styles['Top surface']['color'])
            ))
        if show_bot:
            fig2.add_trace(go.Scatter(
                x=x, y=Tbot, mode='lines+markers', name='Bot surface',
                line=line_styles['Bot surface'],
                marker=dict(symbol=mark_syms['Bot surface'], size=6, color=line_styles['Bot surface']['color'])
            ))
        if show_1d:
            fig2.add_trace(go.Scatter(
                x=x, y=T1, mode='lines+markers', name='1D soln',
                line=line_styles['1D soln'],
                marker=dict(symbol=mark_syms['1D soln'], size=6, color=line_styles['1D soln']['color'])
            ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='x (m)', yaxis_title='Temperature (°C)',
            legend=dict(title='Profile')
        )
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(showgrid=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Temperature Differences
    st.subheader("Temperature Differences")
    dm = st.checkbox("Mid−Top")
    da = st.checkbox("avg−1D")
    if any([dm, da]):
        fig3 = go.Figure()
        if dm:
            fig3.add_trace(go.Scatter(
                x=x, y=Tmid - Ttop, mode='lines+markers', name='Mid−Top',
                line=dict(color='orange', dash='dash'),
                marker=dict(symbol='circle', color='orange', size=6)
            ))
        if da:
            fig3.add_trace(go.Scatter(
                x=x, y=Tavg - T1, mode='lines+markers', name='avg−1D',
                line=dict(color='brown', dash='dot'),
                marker=dict(symbol='square', color='brown', size=6)
            ))
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='x (m)', yaxis_title='ΔT (°C)',
            legend=dict(title='Difference')
        )
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(showgrid=False)
        st.plotly_chart(fig3, use_container_width=True)

    # Download CSV
    df = pd.DataFrame({'x': X.flatten(), 'y': Yg.flatten(), 'T': T2.flatten()})
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download contour data CSV", buf, "temp_contour.csv", "text/csv")

footer()
