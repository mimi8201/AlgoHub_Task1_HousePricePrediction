import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Real Estate AI", 
    page_icon="🏢", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. UI Theme & Custom CSS Injection
# ==========================================

st.markdown("""
    <style>
    /* -----------------------------------
       Base Application Backgrounds
       ----------------------------------- */
    [data-testid="stAppViewContainer"] {
        background-color: #1B262C;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0F4C75;
        min-width: 400px !important;
        max-width: 400px !important;
    }

    /* -----------------------------------
       Typography & Readability
       ----------------------------------- */
    h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown p {
        color: #FFFFFF !important; 
    }

    .main-header {
        font-size: 3.0rem !important; 
        color: #BBE1FA !important;
        font-weight: 900;
        margin-bottom: -10px;
        line-height: 1.2;
    }
    
    .sub-header {
        font-size: 1.3rem !important; 
        color: #3282B8 !important;
        margin-bottom: 2rem;
        font-weight: 600;
    }

    /* -----------------------------------
       Input Fields & Stepper Buttons Fix
       ----------------------------------- */
    div[data-baseweb="input"], 
    div[data-baseweb="base-input"],
    input {
        background-color: #1B262C !important;
        color: #BBE1FA !important;
        border: 1px solid #3282B8 !important;
        border-radius: 4px;
    }
    
    button[aria-label="Step up"], 
    button[aria-label="Step down"],
    [data-testid="stNumberInputStepUp"], 
    [data-testid="stNumberInputStepDown"] {
        background-color: #0F4C75 !important;
        color: #BBE1FA !important;
        border-left: 1px solid #3282B8 !important; 
    }
    
    button[aria-label="Step up"]:hover, 
    button[aria-label="Step down"]:hover {
        background-color: #3282B8 !important;
        color: #FFFFFF !important;
    }

    /* -----------------------------------
       Primary Button
       ----------------------------------- */
    /* Target the button container */
    button[kind="primary"], 
    div.stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #BBE1FA !important; /* Light blue base */
        border: 2px solid #FFFFFF !important; 
        border-radius: 30px !important;       /* Rounded edges */
        padding: 10px !important;
        transition: all 0.3s ease !important; /* Smooth animation */
    }
    
    button[kind="primary"] p, 
    div.stButton > button[kind="primary"] p,
    div[data-testid="stFormSubmitButton"] > button p {
        color: #0F4C75 !important;            /* Dark text normally */
        font-weight: 900 !important;
        font-size: 1.1rem !important;
        transition: color 0.3s ease !important;
    }
    
    /* Hover state for the button background */
    button[kind="primary"]:hover, 
    div.stButton > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #0F4C75 !important; /* Dark blue on hover */
        border-color: #BBE1FA !important;
    }

    /* Hover state for the button text */
    button[kind="primary"]:hover p, 
    div.stButton > button[kind="primary"]:hover p,
    div[data-testid="stFormSubmitButton"] > button:hover p {
        color: #FFFFFF !important;            /* White text on hover */
    }

    /* -----------------------------------
       Metric Component Styling
       ----------------------------------- */
    div[data-testid="metric-container"] {
        background-color: #0F4C75 !important;
        border: 1px solid #3282B8;
        border-top: 6px solid #BBE1FA;
        padding: 25px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }
    
    div[data-testid="stMetricValue"] > div {
        color: #FFFFFF !important;
        font-size: 3.5rem !important;
        font-weight: 900 !important;
    }
    
    div[data-testid="stMetricDelta"] > div {
        color: #BBE1FA !important;
    }

    /* -----------------------------------
       Sidebar Form Container
       ----------------------------------- */
    div[data-testid="stForm"] {
        border-left: 4px solid #BBE1FA;
        background-color: rgba(27, 38, 44, 0.6);
        padding: 1.5rem;
        border-radius: 4px;
    }
    
    /* -----------------------------------
       Expander ("View Backend Analytics")
       ----------------------------------- */
    div[data-testid="stExpander"] {
        background-color: #0F4C75 !important;
        border: 1px solid #3282B8 !important;
        border-radius: 8px !important;
        overflow: hidden; 
    }
    
    div[data-testid="stExpander"] summary {
        background-color: #0F4C75 !important;
        color: #BBE1FA !important;
    }
    
    div[data-testid="stExpanderDetails"] {
        background-color: #1B262C !important; 
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 3. Backend Integration Methods
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/model.pkl')
        zip_map = joblib.load('models/zip_freq_map.pkl')
        return model, zip_map
    except FileNotFoundError:
        st.error("Model artifacts not found. Please verify the 'models' directory exists.")
        return None, None


# ==========================================
# 4. Main Application Loop
# ==========================================
def main():
    st.sidebar.title("⚙️ Property Details")
    st.sidebar.markdown("Adjust the parameters below to update the model's prediction.")
    
    model, zip_map = load_artifacts()
    if model is None: return

    with st.sidebar.form("prediction_form"):
        beds = st.number_input("Bedrooms", min_value=1, max_value=15, value=3, step=1)
        baths = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        size = st.number_input("Size (Sq Ft)", min_value=500, max_value=15000, value=1500, step=100)
        zip_code = st.text_input("Zip Code", value="98103")
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="Generate Valuation", type="primary", use_container_width=True)

    st.markdown('<p class="main-header">Real Estate Valuation Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Machine Learning</p>', unsafe_allow_html=True)
    st.divider()

    if submit_button:
        if not zip_code.isdigit():
            st.error("Data Validation Error: Please enter a valid numeric Zip Code.")
            return
            
        zip_code_int = int(zip_code)
        beds_baths_total = beds + baths
        
        if zip_code_int in zip_map:
            zip_code_freq = zip_map[zip_code_int]
        else:
            st.warning("Unrecognized Zip Code. Applying baseline market frequency.")
            zip_code_freq = 1 

        input_data = pd.DataFrame([[
            beds, baths, size, zip_code_freq, beds_baths_total
        ]], columns=['beds', 'baths', 'size', 'zip_code_freq', 'beds_baths_total'])

        prediction = model.predict(input_data)[0]

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label="Estimated Market Value", 
                value=f"${prediction:,.2f}", 
                delta="Model Prediction"
            )
            
        st.write("") 
        st.write("")
        
        with st.expander("View Backend Analytics"):
            st.markdown("**Engineered Features Passed to Model:**")
            
            st.markdown(f"""
            <div style="background-color: #0F4C75; margin: 15px 5px; padding: 15px; border-radius: 8px; color: #BBE1FA; font-family: monospace; border: 1px dashed #BBE1FA;">
            Total Rooms (Beds + Baths) : {beds_baths_total}<br>
            Zip Code Frequency Score   : {zip_code_freq}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()