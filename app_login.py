import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.optimize import fsolve
# -----------------------------------------------
# Background Styling with CSS
# -----------------------------------------------
def add_background_styles():
    """Add background color styling using CSS."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: rgb(255, 255, 255);
            color: black;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            color: black;
            font-size: large;
        }
         
         label {
            color: black !important;  /* Make labels (e.g., 'Username' and 'Password') black */
        }
         button {
            color: red !important; /* Ensure button text is red */
        }
    
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------
# Login Function
# -----------------------------------------------
def login():


    # Display the logo only on the login page
    logo_path = "MEEN_logo.png"  # Ensure the image file is in the same directory
    logo = Image.open(logo_path)
    st.image(logo, width=400, use_column_width="never")  # Adjust the width as needed
    st.title("Login")
    # Username and password inputs
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Predefined credentials
        credentials = {
            "Guest": "GuestPass",
            "Aditya": "Yalamanchili",
            "Prabhakar": "Pagilla",
            "admin": "adminpass"
        }
        if username in credentials and credentials[username] == password:
            st.session_state.logged_in = True
            
        else:
            st.error("Invalid username or password. Please try again.")

# -----------------------------------------------
# Footer
# -----------------------------------------------
def display_footer():
    st.markdown(
        """
        <div class="footer">
            Version α 0.1 | © 2025 Texas A&M University
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------
# Main Application
# -----------------------------------------------
if __name__ == "__main__":
    # Apply background styling globally
    add_background_styles()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.session_state.logged_in = login()
    else:
        def temperature_distribution(k, rho, c, v, T0, T_inf, h, Y, L, n_max):
            """
            Returns (X, Y_grid, T) where:
              X, Y_grid = meshgrid of x and y
              T         = temperature distribution (2D array)
             
            """
            from scipy.optimize import fsolve
            import numpy as np
            
            # Calculate beta and Bi
            beta = rho * c * v / (2.0 * k)
            Bi = h * Y / k
            
            # Define the equation f(lambda) = lambda * Y * tan(lambda * Y) - Bi
            def f_lambda(lmbd):
                return lmbd * Y * np.tan(lmbd * Y) - Bi
        
            # Find eigenvalues lambda_n
            lambda_n = []
            for n in range(n_max+1):
                lambda_guess = (n + Y) * np.pi / Y
                sol = fsolve(f_lambda, lambda_guess)
                lambda_n.append(sol[0])
            lambda_n = np.array(lambda_n)
        
            # Calculate a_n
            a_n = np.zeros(n_max+1)
            for i in range(0, n_max):
                lam = lambda_n[i]
                numerator = 2.0 * (T0 - T_inf) * np.sin(lam * Y)
                denominator = lam * Y + np.sin(lam * Y) * np.cos(lam * Y)
                a_n[i] = numerator / denominator
        
            # Create spatial domain
            x_vals = np.linspace(0, L, 500)
            y_vals = np.linspace(-Y, Y, 200)
            X, Y_grid = np.meshgrid(x_vals, y_vals)
        
            # Calculate Theta(x, y)
            Theta = np.zeros_like(X)
            for i in range(0, n_max):
                lam = lambda_n[i]
                exponent_term = (beta - np.sqrt(beta**2 + lam**2)) * X
                Theta += a_n[i] * np.exp(exponent_term) * np.cos(lam * Y_grid)
        
            # Convert Theta to actual temperature
            T = Theta + T_inf
        
            return X, Y_grid, T


        
        st.title("Web temperature distribution simulator tool")

        st.sidebar.header("Input Parameters")
        # --- New: Boundary Condition Dropdown ---
        bc_option = st.sidebar.selectbox(
            "Boundary Condition",
            [
                "Free span convective cooling", 
                "Web over a heated/ cooled roller", 
                "Web in a heating/ cooling zone"
            ]
        )
        
        k = st.sidebar.number_input("k (W/m-K)", value=0.23, format="%.5f")
        rho = st.sidebar.number_input("rho (kg/m^3)", value=1390.0, format="%.1f")
        c = st.sidebar.number_input("c (J/kg-K)", value=1400.0, format="%.1f")
        v = st.sidebar.number_input("v (m/s)", value=0.1, format="%.3f")
        T0 = st.sidebar.number_input("T0 (initial temp, °C)", value=100.0, format="%.1f")
        T_inf = st.sidebar.number_input("T_inf (ambient temp, °C)", value=25.0, format="%.1f")
        h = st.sidebar.number_input("h (W/m^2-K)", value=500.0, format="%.1f")
        Y = st.sidebar.number_input("Y (half-thickness, m)", value=0.0005, format="%.5f")
        L = st.sidebar.number_input("L (length, m)", value=1.0, format="%.2f")
        n_max = st.sidebar.number_input("Number of eigenvalues (n_max)", value=10,
                                        min_value=1, max_value=50, step=1)

         # Display the selected boundary condition image on the main page
        st.subheader("Selected Boundary Condition")
        if bc_option == "Free span convective cooling":
            st.image("BC2.png", caption="Free Span Convective Cooling", width=500,use_column_width="auto")
        elif bc_option == "Web over a heated/ cooled roller":
            st.image("BC1.png", caption="Web over a Heated/Cooled Roller", width=500)
        else:  # "Web in a heating/ cooling zone"
            st.image("BC4.png", caption="Web in a Heating/Cooling Zone", width=500)
        
        if st.button("Compute Temperature Distribution"):
            with st.spinner("Computing..."):
                X, Y_grid, T = temperature_distribution(
                    k, rho, c, v, T0, T_inf, h, Y, L, n_max
                )

            # Plot the temperature distribution
            st.subheader("Contour Plot of Temperature")
            fig, ax = plt.subplots(figsize=(8,5))
            contour = ax.contourf(X, Y_grid, T, 100, cmap='turbo', vmin=0, vmax=100)
            plt.colorbar(contour, ax=ax, label='Temperature (°C)')
            ax.set_title("Steady-state temperature distribution in the moving web")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            st.pyplot(fig)

            # --- 2) Temperature Profiles at y=0 (mid-plane) and y=Y (top surface) ---
            st.subheader("Temperature Profiles at Mid-plane (y=0) and Top Surface (y=+Y)")

            # Spatial arrays for plotting along x
            x_vals = X[0, :]      # shape (500,)
            y_vals = Y_grid[:, 0] # shape (200,)

            # Find the indices for y=0 and y=+Y
            mid_plane_idx = np.argmin(np.abs(y_vals - 0))
            top_surface_idx = np.argmin(np.abs(y_vals - (Y)))

            # Extract temperature at these two rows
            T_mid_plane = T[mid_plane_idx, :]    # shape (500,)
            T_top_surface = T[top_surface_idx, :] # shape (500,)

            fig2, ax2 = plt.subplots(figsize=(8,4))
            ax2.plot(x_vals, T_mid_plane, label='y=0 (Mid-plane)')
            ax2.plot(x_vals, T_top_surface, label=f'y={Y:.6f} (Top Surface)')
            ax2.set_xlabel("x (m)")
            ax2.set_ylabel("Temperature (°C)")
            ax2.set_title("Temperature Profiles")
            ax2.legend()
            st.pyplot(fig2)




            

            # Optional: Download temperature data
            st.subheader("Download Temperature Data as CSV")
            nx = X.shape[1]
            ny = X.shape[0]
            X_flat = X.flatten()
            Y_flat = Y_grid.flatten()
            T_flat = T.flatten()

            df = pd.DataFrame({
                'x_m': X_flat,
                'y_m': Y_flat,
                'Temperature_C': T_flat
            })

            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            st.download_button(
                label="Download CSV",
                data=csv_buffer,
                file_name="temperature_distribution.csv",
                mime="text/csv"
            )

    # Persistent footer
    display_footer()
