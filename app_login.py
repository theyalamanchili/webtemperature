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
# Persistent Footer
# -----------------------------------------------
def footer():
    st.markdown(
        """
        <div class='footer'>Version α 0.1 | © 2025 Texas A&amp;M University</div>
        """, unsafe_allow_html=True)

# -----------------------------------------------
# 2D / 1D Solvers
# -----------------------------------------------

def solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N):
    Y = t / 2
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
        (2*dT * np.sin(z[i])) / (z[i] + np.sin(z[i]) * np.cos(z[i]))
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
    T2 = Tinf + Theta
    return x, y, X, Yg, T2


def solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x):
    Y = t / 2
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

# Init login state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Show login until done
if not st.session_state.logged_in:
    login()
    st.stop()

# Main UI
st.title("Web Temperature Distribution Simulator")

# Boundary Condition
st.sidebar.header("Boundary Condition")
bc = st.sidebar.selectbox(
    "Select boundary condition:",
    ["Free span convective cooling", "Web over heated/cooled roller", "Web in heating/cooling zone"]
)
if bc == "Free span convective cooling":
    st.image("BC2.png", caption="Free span convective cooling", use_column_width=True)
    st.sidebar.markdown("_Free span: convective cooling both sides._")
elif bc == "Web over heated/cooled roller":
    st.image("BC1.png", caption="Web over heated/cooled roller", use_column_width=True)
    st.sidebar.markdown("_Contact with roller at fixed T._")
else:
    st.image("BC3.png", caption="Web in heating/cooling zone", use_column_width=True)
    st.sidebar.markdown("_Traveling through heating/cooling zone._")

# Material Properties
st.sidebar.header("Material Properties")
matlib = {'Aluminum': {'k':237,'rho':2700,'c':897},
          'Copper':   {'k':401,'rho':8960,'c':385},
          'PET':      {'k':0.2,'rho':1390,'c':1400}}
mat = st.sidebar.selectbox("Material", list(matlib.keys())+['Custom'])
if mat!='Custom':
    k,rho,c = matlib[mat].values()
    st.sidebar.write(f"k={k}, rho={rho}, c={c}")
else:
    k   = st.sidebar.number_input("k [W/m·K]",0.1,500.0,0.2)
    rho = st.sidebar.number_input("rho [kg/m³]",100,20000,1390)
    c   = st.sidebar.number_input("c [J/kg·K]",100,5000,1400)

# Process Parameters
st.sidebar.markdown("---")
st.sidebar.header("Process Parameters")
v    = st.sidebar.number_input("v [m/s]",0.01,10.0,1.6)
T0   = st.sidebar.number_input("T₀ [°C]",-50.0,500.0,200.0)
Tinf = st.sidebar.number_input("T∞ [°C]",-50.0,200.0,25.0)
h    = st.sidebar.number_input("h [W/m²·K]",1.0,10000.0,100.0)
t    = st.sidebar.number_input("thickness t [m]",1e-6,1e-2,0.001,step=1e-6,format="%.6f")
W    = st.sidebar.number_input("Width W [m]",0.01,5.0,1.0)
L    = st.sidebar.number_input("Span L [m]",0.1,50.0,10.0)
N    = st.sidebar.slider("Terms N",5,50,20)

# Compute
if st.button("Compute"):
    if bc=="Free span convective cooling":
        x,y,X,Yg,T2 = solve_2d(k,rho,c,v,T0,Tinf,h,t,L,N)
        T1        = solve_1d(k,rho,c,v,T0,Tinf,h,t,W,x)
        st.session_state.update(x=x,y=y,X=X,Yg=Yg,T2=T2,T1=T1)
        st.session_state.ready=True
    else:
        st.warning("Solver coming soon.")

# Display results
if st.session_state.get('ready',False):
    x,y,X,Yg,T2,T1 = (st.session_state[k] for k in ['x','y','X','Yg','T2','T1'])
    # Biot & Pe
    Yh    = t/2
    Bi_num= h*Yh/k
    Pe_num= v*L/(k/(rho*c))
    # Contour
    st.subheader("2D Temperature Contour")
    show = st.checkbox("Show contour lines & labels")
    fig = go.Figure(go.Contour(
        z=T2,x=x,y=y,colorscale='Turbo',ncontours=60,
        contours=dict(showlines=show,showlabels=show,
                      labelfont=dict(size=12,color='black'))
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,use_container_width=True)
    # Bi/Pe display
    st.markdown(f"""
**Biot number (Bi):** {Bi_num:.2f}  
**Péclet number (Pe):** {Pe_num:.1f}
"""
    )
    st.markdown(
        """
        Bi: surface/conduction; low Bi→cond‑dominated.  
        Pe: advection/conduction; high Pe→adv‑dominated.
        """
    )
    # Profiles
    st.subheader("Temperature Profiles")
    idx_mid=np.argmin(np.abs(y))
    idx_top=np.argmin(np.abs(y-Yh))
    idx_bot=np.argmin(np.abs(y+Yh))
    Tavg=T2.mean(axis=0); Tmid=T2[idx_mid]; Ttop=T2[idx_top]; Tbot=T2[idx_bot]
    styles={'avg':{'color':'blue','dash':'solid'},
            'mid':{'color':'green','dash':'dash'},
            'top':{'color':'red','dash':'dot'},
            'bot':{'color':'purple','dash':'dashdot'},
            '1d':{'color':'black','dash':'longdash'}}
    marks={'avg':'circle','mid':'square','top':'triangle-up','bot':'triangle-down','1d':'x'}
    s_avg=st.checkbox("2D avg"); s_mid=st.checkbox("Mid"); s_top=st.checkbox("Top"); s_bot=st.checkbox("Bot"); s_1d=st.checkbox("1D")
    if any([s_avg,s_mid,s_top,s_bot,s_1d]):
        fig2=go.Figure()
        if s_avg: fig2.add_trace(go.Scatter(x=x,y=Tavg,mode='lines+markers',name='2D avg',line=styles['avg'],marker=dict(symbol=marks['avg'],color=styles['avg']['color'])))
        if s_mid: fig2.add_trace(go.Scatter(x=x,y=Tmid,mode='lines+markers',name='Mid',line=styles['mid'],marker=dict(symbol=marks['mid'],color=styles['mid']['color'])))
        if s_top: fig2.add_trace(go.Scatter(x=x,y=Ttop,mode='lines+markers',name='Top',line=styles['top'],marker=dict(symbol=marks['top'],color=styles['top']['color'])))
        if s_bot: fig2.add_trace(go.Scatter(x=x,y=Tbot,mode='lines+markers',name='Bot',line=styles['bot'],marker=dict(symbol=marks['bot'],color=styles['bot']['color'])))
        if s_1d: fig2.add_trace(go.Scatter(x=x,y=T1,mode='lines+markers',name='1D',line=styles['1d'],marker=dict(symbol=marks['1d'],color=styles['1d']['color'])))
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis_title='x (m)',yaxis_title='T (°C)',legend=dict(title='Profiles'))
        st.plotly_chart(fig2,use_container_width=True)
    # Differences
    st.subheader("Temperature Differences")
    d_mt=st.checkbox("Mid−Top"); d_avg=st.checkbox("avg−1D")
    if any([d_mt,d_avg]):
        fig3=go.Figure()
        if d_mt: fig3.add_trace(go.Scatter(x=x,y=Tmid-Ttop,mode='lines+markers',name='Mid−Top',line=dict(color='orange',dash='dash'),marker=dict(symbol='circle',color='orange')))
        if d_avg:fig3.add_trace(go.Scatter(x=x,y=Tavg-T1,mode='lines+markers',name='avg−1D',line=dict(color='brown',dash='dot'),marker=dict(symbol='square',color='brown')))
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis_title='x (m)',yaxis_title='ΔT (°C)',legend=dict(title='Diffs'))
        st.plotly_chart(fig3,use_container_width=True)
    # Download
    df=pd.DataFrame({'x':X.flatten(),'y':Yg.flatten(),'T':T2.flatten()})
    buf=BytesIO();df.to_csv(buf,index=False);buf.seek(0)
    st.download_button("Download CSV",buf,"temp_contour.csv","text/csv")

footer()
