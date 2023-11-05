# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from PIL import Image

# Define file paths for the model and data
file_path = "C:/Users/afiqn/OneDrive - Universiti Teknologi PETRONAS/Desktop/UTP/FYP/House-Rental-Price-Prediction-main/model.pkl"
model1 = joblib.load(file_path)
mean1 = np.load("C:/Users/afiqn/OneDrive - Universiti Teknologi PETRONAS/Desktop/UTP/FYP/House-Rental-Price-Prediction-main/mean.npy")
std1 = np.load("C:/Users/afiqn/OneDrive - Universiti Teknologi PETRONAS/Desktop/UTP/FYP/House-Rental-Price-Prediction-main/std.npy")

# 1. Vertical sidebar
# Create a Streamlit sidebar with options
with st.sidebar:
     # User selects between Rental Price Prediction and Rental Market Analysis
    selected = option_menu(
        menu_title="Rental Price Prediction Tool",
        options=['Rental Price Prediction','Rental Market Analysis'],
        icons=["house", "book"],
        menu_icon="cash-stack",
        default_index=0
     )

# # 2. Horizontal
# selected = option_menu(
#         menu_title="Rental Price Prediction Tool",
#         options=['Rental Price Prediction','Rental Market Analysis'],
#         icons=["house", "book"],
#         menu_icon="cash-stack",
#         default_index=0,
#         orientation="horizontal",
#         # styles={
#         #         "container": {"padding": "1!important", "background-color": "#4b637d"},
#         #         "icon": {"color": "orange", "font-size": "25px"},
#         #         "nav-link": {
#         #             "font-size": "25px",
#         #             "text-align": "left",
#         #             "margin": "0px",
#         #             "--hover-color": "#69092f",
#         #         },
#         #         "nav-link-selected": {"background-color": "#db0b5e"},
#         #     },
#     )

# Prediction
if(selected == 'Rental Price Prediction'):
    # Mapping for the 'furnished' feature to numerical values
    furnished_mapping = {'Not furnished': 0, 'Partially furnished': 1, 'Fully furnished': 2}
    # Mapping for the 'nearby_railways' feature to numerical values
    nearby_railways_mapping = {'Yes': 1, 'No': 0}
    # Mapping for the 'minimart_availability' feature to numerical values
    minimart_availability_mapping = {'Yes': 1, 'No': 0}
    # Mapping for the 'security_availability' feature to numerical values
    security_availability_mapping = {'Yes': 1, 'No': 0}
    # Mapping for the 'property_type' feature to numerical values
    # It's a nested dictionary where each property type is mapped to a dictionary of subtypes with binary values
    property_type_mapping = {
        'Condominium': {'Condominium': 1, 'Apartment': 0, 'Service Residence': 0, 'Others': 0},
        'Apartment': {'Condominium': 0, 'Apartment': 1, 'Service Residence': 0, 'Others': 0},
        'Service Residence': {'Condominium': 0, 'Apartment': 0, 'Service Residence': 1, 'Others': 0},
        'Others': {'Condominium': 0, 'Apartment': 0, 'Service Residence': 0, 'Others': 1}
    }

    # Custom banner
    image = Image.open('prop9.png')
    st.image(image)

    # Background colour
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"]{
    background: linear-gradient(180deg, #27140c, #57392c, #78645b)
    }

    [data-testid="stSidebar"]{
    background-color: #272727;
    opacity: 0.8;
    background-image:  linear-gradient(30deg, #000000 12%, transparent 12.5%, transparent 87%, #000000 87.5%, #000000), linear-gradient(150deg, #000000 12%, transparent 12.5%, transparent 87%, #000000 87.5%, #000000), linear-gradient(30deg, #000000 12%, transparent 12.5%, transparent 87%, #000000 87.5%, #000000), linear-gradient(150deg, #000000 12%, transparent 12.5%, transparent 87%, #000000 87.5%, #000000), linear-gradient(60deg, #00000077 25%, transparent 25.5%, transparent 75%, #00000077 75%, #00000077), linear-gradient(60deg, #00000077 25%, transparent 25.5%, transparent 75%, #00000077 75%, #00000077);
    background-size: 48px 84px;
    background-position: 0 0, 0 0, 24px 42px, 24px 42px, 0 0, 24px 42px;
    }
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Create a Streamlit title for the prediction web application
    st.title("Built-up Rental Price Prediction")

    # Input fields for user to enter property details
    rooms = st.text_input('Number of rooms', placeholder='Enter the number of rooms') 
    parking = st.text_input('Number of parking', placeholder='Enter the number of parking lot(s)')
    bathroom = st.text_input('Number of bathroom', placeholder='Enter the number of bathrooms')
    size = st.text_input('Size in square feet', placeholder='Enter the prefered floor plan size (sq ft.)')
    # Dropdown selection for the furnishing option
    furnished = st.selectbox('Furnishing', ['Not furnished', 'Partially furnished', 'Fully furnished'])
    # Dropdown selection for nearby KTM or LRT availability
    nearKTMLRT = st.selectbox('Nearby KTM or LRT', ['Yes', 'No'])
    # Dropdown selection for minimart availability
    minimart_availability = st.selectbox('Minimart availability', ['Yes', 'No'])
    # Dropdown selection for security availability
    security_availability = st.selectbox('Security availability', ['Yes', 'No'])
    # Dropdown selection for the desired location (Can be left empty if no preferred locations)
    location_name = st.selectbox('Desired location', ['Ampang', 'Ara Damansara', 'Bangi', 'Bangsar South', 'Batu Caves', 'Bukit Jalil', 'Cheras', 'Cyberjaya', 'Damansara Damai', 'Damansara Perdana', 'Dengkil', 'Desa Pandan', 'Gombak', 'Jalan Ipoh', 'Jalan Kuching', 'KL City', 'KLCC', 'Kajang', 'Kepong', 'Keramat', 'Klang', 'Kota Damansara', 'Kuchai Lama', 'Mont Kiara', 'Old Klang Road', 'Other', 'Petaling Jaya', 'Puchong', 'Segambut', 'Selayang', 'Semenyih', 'Sentul', 'Sepang', 'Seri Kembangan', 'Setapak', 'Setia Alam', 'Shah Alam', 'Subang Jaya', 'Sungai Besi', 'Taman Desa', 'Wangsa Maju'])
    # Dropdown selection for the property type
    property_type = st.selectbox('Property type', ['Apartment', 'Condominium', 'Service Residence', 'Others'])
    # Dropdown selection for the region
    region = st.selectbox('Region', ['Selangor', 'Kuala Lumpur'])

    # Convert categorical variables to numerical format
    furnished_value = furnished_mapping[furnished]
    nearby_railways_value = nearby_railways_mapping[nearKTMLRT]
    minimart_availability_value = minimart_availability_mapping[minimart_availability]
    security_availability_value = security_availability_mapping[security_availability]
    property_type_value = property_type_mapping[property_type]

    # Set all location columns to 0 initially
    location_columns = {'Cheras', 'Setapak', 'Sentul', 'Kepong', 'Bukit Jalil', 'Ampang', 'Wangsa Maju', 'Old Klang Road', 'Taman Desa', 'Mont Kiara','Keramat','Jalan Ipoh','Sungai Besi','Kuchai Lama','KL City','Jalan Kuching','Segambut','Desa Pandan','KLCC','Bangsar South','Cyberjaya', 'Kajang', 'Puchong', 'Seri Kembangan', 'Shah Alam', 'Petaling Jaya', 'Semenyih', 'Subang Jaya', 'Setia Alam', 'Bangi','Damansara Perdana','Batu Caves','Damansara Damai','Sepang','Kota Damansara','Klang','Selayang','Gombak','Dengkil','Ara Damansara','Other'}

    # Check if the entered location matches any location in the location list
    if location_name in location_columns:
        # Set the selected location to 1 and all other locations to 0
        location_columns = {location: 1 if location == location_name else 0 for location in location_columns}
    else:
        # If no match, set 'Others' to 1 and all locations to 0
        location_columns = {location: 0 for location in location_columns}
        location_columns['Other'] = 1

    size_float = float(size) if size else 0
    rooms_int = int(rooms) if rooms else 0
    parking_int = int(parking) if parking else 0
    bathroom_int = int(bathroom) if bathroom else 0
    selangor_int = 1 if region == 'Selangor' else 0
    kuala_lumpur_int = 1 if region == 'Kuala Lumpur' else 0

    # Create the user inputs DataFrame
    user_inputs = {
        'rooms': [rooms_int],
        'parking': [parking_int],
        'bathroom': [bathroom_int],
        'size(sq.ft.)': [size_float],
        'furnished': [furnished_value],
        'near_KTM-LRT': [nearby_railways_value],
        'minimart_availability': [minimart_availability_value],
        'security_availability': [security_availability_value],
        **location_columns,  # Include all location columns in the dictionary
        **property_type_value,
        'Selangor': [selangor_int],
        'Kuala Lumpur': [kuala_lumpur_int]
    }

    user_inputs_df = pd.DataFrame(user_inputs)

    if st.button('Predict price'):
        # Scale the input data by user using the mean and standard deviation
        user_inputs_scaled = (user_inputs_df - mean1) / std1
        
        # Check if the input data has the correct number of features
        if user_inputs_scaled.shape[1] != len(mean1):
            st.error(f'Invalid number of features. Expected {len(mean1)} features, but received {user_inputs_scaled.shape[1]}')
        else:
            # Make a prediction using the machine learning model
            price_pred = model1.predict(user_inputs_scaled.values.reshape(1, -1))
            # Display a label for the predicted price
            st.markdown(
                f'<div style="font-size: 30px; color: #FFFFFF;">Rental Price:</div>',
                unsafe_allow_html=True
            )
            predicted_price = f"RM {price_pred[0]:,.2f} / month"

            # Format and display the predicted price with styling
            st.markdown(
                f'<div style="background-color:#432E28;padding:20px;border-radius:10px;font-size:24px;color:white;box-shadow: 3px 3px 10px 1px rgba(0,0,0,0.75);">{predicted_price}</div>',
                unsafe_allow_html=True,
            )

# Analysis
if selected == 'Rental Market Analysis':
    # Load data for market analysis
    df = pd.read_csv("C:/Users/afiqn/OneDrive - Universiti Teknologi PETRONAS/Desktop/UTP/FYP/House-Rental-Price-Prediction-main/cleaned-mudah-apartment-kl-selangor.csv")
    
    # Custom banner
    image = Image.open('prop10.png')
    st.image(image)

    #Background colour
    page_bg_img = """
        <style>
        [data-testid="stAppViewContainer"]{
        background-colour: #0E1117)
        }

        [data-testid="stSidebar"]{
        background-color: #272727;
        opacity: 0.8;
        background-image:  linear-gradient(30deg, #000000 12%, transparent 12.5%, transparent 87%, #000000 87.5%, #000000), linear-gradient(150deg, #000000 12%, transparent 12.5%, transparent 87%, #000000 87.5%, #000000), linear-gradient(30deg, #000000 12%, transparent 12.5%, transparent 87%, #000000 87.5%, #000000), linear-gradient(150deg, #000000 12%, transparent 12.5%, transparent 87%, #000000 87.5%, #000000), linear-gradient(60deg, #00000077 25%, transparent 25.5%, transparent 75%, #00000077 75%, #00000077), linear-gradient(60deg, #00000077 25%, transparent 25.5%, transparent 75%, #00000077 75%, #00000077);
        background-size: 48px 84px;
        background-position: 0 0, 0 0, 24px 42px, 24px 42px, 0 0, 24px 42px;
        }
        </style>
        """

    st.markdown(page_bg_img, unsafe_allow_html=True)
    # Display top features affecting rental price
    st.markdown("## Top 5 Features Affecting the Rental Price based on Feature Importance:")
    st.write("1. Size")
    st.write("2. Furnishing level")
    st.write("3. Number of bathrooms")
    st.write("4. Number of rooms")
    st.write("5. Number of parking")
    
    # Plot average property price by region
    # filtered_df = df[df['region'].isin(['Kuala Lumpur', 'Selangor'])]
    # region_avg_price = filtered_df.groupby('region')['monthly_rent'].mean()
    # fig1, ax1 = plt.subplots()
    # region_avg_price.plot(kind='bar', ax=ax1)
    # ax1.set_xlabel('Region')
    # ax1.set_ylabel('Average Price')
    # ax1.set_title('Average Property Price: Kuala Lumpur vs Selangor')
    # st.pyplot(fig1)

    # AVERAGE PROPERTY PRICE BY REGION
    # Filter data for Kuala Lumpur and Selangor
    filtered_df = df[df['region'].isin(['Kuala Lumpur', 'Selangor'])]
    region_avg_price = filtered_df.groupby('region')['monthly_rent'].mean()
    # Create a Matplotlib bar chart with customized colors, transparent background, and white fonts
    fig, ax = plt.subplots()
    colors = ['purple', 'green']  # Specify the colors for the bars
    region_avg_price.plot(kind='bar', ax=ax, color=colors)
    ax.set_xlabel('Region')
    ax.set_ylabel('Average Price')
    st.title('Average Property Price')
    # Set the background color to be transparent and font color to white
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    # Display the Matplotlib chart in Streamlit
    st.pyplot(fig)

    # Add a gap or spacing between the charts using st.markdown

    # Filter data for Kuala Lumpur and Selangor
    filtered_df = df[df['region'].isin(['Kuala Lumpur', 'Selangor'])]
    region_avg_price = filtered_df.groupby('region')['monthly_rent'].mean()
    # Create a Streamlit bar chart with title and description
    st.title('Average Property Price: Kuala Lumpur vs Selangor')
    st.bar_chart(region_avg_price, use_container_width=True)
    
    
    #PROPERTY SIZE BY REGION
    # Filter data for Kuala Lumpur and Selangor
    kl_df = df[df['region'] == 'Kuala Lumpur']
    selangor_df = df[df['region'] == 'Selangor']
    kl_sizes = kl_df['size(sq.ft.)']
    selangor_sizes = selangor_df['size(sq.ft.)']
    num_bins = 20
    # Create a Matplotlib histogram with customized colors, transparent background, and white fonts
    fig1, ax1 = plt.subplots()
    ax1.hist(kl_sizes, bins=num_bins, alpha=0.5, label='Kuala Lumpur', color='purple')  # Specify the color
    ax1.hist(selangor_sizes, bins=num_bins, alpha=0.5, label='Selangor', color='green')  # Specify the color
    ax1.set_xlabel('Property Size')
    ax1.set_ylabel('Count of Properties')
    st.title('Property Size Distribution by Region')
    ax1.legend()
    # Set the background color to be transparent and font color to white
    fig1.patch.set_alpha(0)
    ax1.set_facecolor('none')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.yaxis.label.set_color('white')
    ax1.xaxis.label.set_color('white')
    ax1.title.set_color('white')
    # Display the Matplotlib chart in Streamlit
    st.pyplot(fig1)


    #FURNISHING
    # Calculate counts for each type of furnishing
    no_furniture_count = df[df['furnished'] == 0].shape[0]
    partial_furniture_count = df[df['furnished'] == 1].shape[0]
    fully_furnished_count = df[df['furnished'] == 2].shape[0]
    # Create data for the donut chart
    labels = ['No Furniture', 'Partially Furnished', 'Fully Furnished']
    sizes = [no_furniture_count, partial_furniture_count, fully_furnished_count]
    # Create a Matplotlib donut chart with a transparent background
    fig2, ax2 = plt.subplots()
    wedges, texts, autotexts = ax2.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='w'),
    )
    # Add boxes behind the data labels and percentages
    for t in texts:
        t.set(size=14, weight='bold', color='w')
        t.set_bbox(dict(facecolor='#8B4513', edgecolor='w', boxstyle='round,pad=0.2'))
    for at in autotexts:
        at.set(size=14, weight='bold', color='w')
    ax2.axis('equal')
    st.title('Proportion of Furnished Properties')
    # Set the background color to be transparent
    fig2.patch.set_alpha(0)
    ax2.set_facecolor('none')
    ax2.title.set_color('white')
    # Display the Matplotlib donut chart in Streamlit
    st.pyplot(fig2)


    #PROPERTY BATHROOMS BY REGION
    # Plot property bathrooms by region
    # Filter data for Kuala Lumpur and Selangor
    kl_df = df[df['region'] == 'Kuala Lumpur']
    selangor_df = df[df['region'] == 'Selangor']
    bathroom_categories = sorted(list(set(kl_df['bathroom'].unique()) | set(selangor_df['bathroom'].unique())))
    kl_bathroom_counts = kl_df['bathroom'].value_counts().sort_index().reindex(bathroom_categories, fill_value=0)
    selangor_bathroom_counts = selangor_df['bathroom'].value_counts().sort_index().reindex(bathroom_categories, fill_value=0)
    # Create a Matplotlib bar chart with a transparent background, purple for KL, and green for Selangor
    fig3, ax3 = plt.subplots()
    bar_width = 0.35
    x = range(len(bathroom_categories))
    ax3.bar(x, kl_bathroom_counts, width=bar_width, label='Kuala Lumpur', color='purple')
    ax3.bar(x, selangor_bathroom_counts, width=bar_width, label='Selangor', bottom=kl_bathroom_counts, color='green')
    ax3.set_xlabel('Number of bathrooms', color='white')  # Set the axis title color to white
    ax3.set_ylabel('Count of Properties', color='white')  # Set the axis title color to white
    st.title('Property Bathrooms by Region')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bathroom_categories)
    ax3.legend()
    # Set the background color to be transparent
    fig3.patch.set_alpha(0)
    ax3.set_facecolor('none')
    ax3.title.set_color('white')
    # Set the axis and ticks to white
    ax3.spines['bottom'].set_color('white')
    ax3.spines['top'].set_color('white')
    ax3.spines['right'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.tick_params(axis='x', colors='white')
    ax3.tick_params(axis='y', colors='white')
    # Display the Matplotlib chart in Streamlit
    st.pyplot(fig3)


    #PROPERTY ROOMS BY REGION
    # Plot property rooms by region
    # Filter data for Kuala Lumpur and Selangor
    kl_df = df[df['region'] == 'Kuala Lumpur']
    selangor_df = df[df['region'] == 'Selangor']
    room_categories = sorted(list(set(kl_df['rooms'].unique()) | set(selangor_df['rooms'].unique())))
    kl_room_counts = kl_df['rooms'].value_counts().sort_index().reindex(room_categories, fill_value=0)
    selangor_room_counts = selangor_df['rooms'].value_counts().sort_index().reindex(room_categories, fill_value=0)
    # Create a Matplotlib bar chart with a transparent background, purple for KL, and green for Selangor
    fig4, ax4 = plt.subplots()
    bar_width = 0.35
    x = range(len(room_categories))
    ax4.bar(x, kl_room_counts, width=bar_width, label='Kuala Lumpur', color='purple')
    ax4.bar(x, selangor_room_counts, width=bar_width, label='Selangor', bottom=kl_room_counts, color='green')
    ax4.set_xlabel('Number of Rooms', color='white')  # Set the axis title color to white
    ax4.set_ylabel('Count of Properties', color='white')  # Set the axis title color to white
    st.title('Property Rooms by Region')
    ax4.set_xticks(x)
    ax4.set_xticklabels(room_categories)
    ax4.legend()
    # Set the background color to be transparent
    fig4.patch.set_alpha(0)
    ax4.set_facecolor('none')
    ax4.title.set_color('white')
    # Set the axis and ticks to white
    ax4.spines['bottom'].set_color('white')
    ax4.spines['top'].set_color('white')
    ax4.spines['right'].set_color('white')
    ax4.spines['left'].set_color('white')
    ax4.tick_params(axis='x', colors='white')
    ax4.tick_params(axis='y', colors='white')
    # Display the Matplotlib chart in Streamlit
    st.pyplot(fig4)


    #PROPERTY PARKING BY REGION
    # Plot property parking by region
    # Filter data for Kuala Lumpur and Selangor
    kl_df = df[df['region'] == 'Kuala Lumpur']
    selangor_df = df[df['region'] == 'Selangor']
    parking_categories = sorted(list(set(kl_df['parking'].unique()) | set(selangor_df['parking'].unique())))
    kl_parking_counts = kl_df['parking'].value_counts().sort_index().reindex(parking_categories, fill_value=0)
    selangor_parking_counts = selangor_df['parking'].value_counts().sort_index().reindex(parking_categories, fill_value=0)
    # Create a Matplotlib bar chart with a transparent background, purple for KL, and green for Selangor
    fig5, ax5 = plt.subplots()
    bar_width = 0.35
    x = range(len(parking_categories))
    ax5.bar(x, kl_parking_counts, width=bar_width, label='Kuala Lumpur', color='purple')
    ax5.bar(x, selangor_parking_counts, width=bar_width, label='Selangor', bottom=kl_parking_counts, color='green')
    ax5.set_xlabel('Number of Parking', color='white')  # Set the axis title color to white
    ax5.set_ylabel('Count of Properties', color='white')  # Set the axis title color to white
    st.title('Property Parking by Region') 
    ax5.set_xticks(x)
    ax5.set_xticklabels(parking_categories)
    ax5.legend()
    # Set the background color to be transparent
    fig5.patch.set_alpha(0)
    ax5.set_facecolor('none')
    ax5.title.set_color('white')
    # Set the axis and ticks to white
    ax5.spines['bottom'].set_color('white')
    ax5.spines['top'].set_color('white')
    ax5.spines['right'].set_color('white')
    ax5.spines['left'].set_color('white')
    ax5.tick_params(axis='x', colors='white')
    ax5.tick_params(axis='y', colors='white')
    # Display the Matplotlib chart in Streamlit
    st.pyplot(fig5)


    #PROPERTY TYPE BY REGION
    # Plot property type by region
    # Create a Matplotlib bar chart with a transparent background, and set the color for the bars
    property_counts = df.groupby(['region', 'property_type']).size().unstack()
    fig6, ax6 = plt.subplots()
    property_counts.plot(kind='bar', stacked=True, ax=ax6)
    # Set the axis title colors to white
    ax6.set_xlabel('Region', color='white')
    ax6.set_ylabel('Count of Properties', color='white')
    st.title('Property Type by Region') 
    # Set the background color to be transparent
    fig6.patch.set_alpha(0)
    ax6.set_facecolor('none')
    ax6.title.set_color('white')
    # Set the axis and ticks to white
    ax6.spines['bottom'].set_color('white')
    ax6.spines['top'].set_color('white')
    ax6.spines['right'].set_color('white')
    ax6.spines['left'].set_color('white')
    ax6.tick_params(axis='x', colors='white')
    ax6.tick_params(axis='y', colors='white')
    # Display the Matplotlib chart in Streamlit
    st.pyplot(fig6)

    