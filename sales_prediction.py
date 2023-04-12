"""
@author: Sanusi Lamido
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder


import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models
model = pickle.load(open('black_friday_model.pkl', 'rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Sales Prediction System', #Title of the OptionMenu
                          
                          ['Black Friday Sales Prediction','Financial Inclusion'], #You can add more options to the sidebar
                          icons=['shop', 'cash'], #BootStrap Icons - Add more depending on the number of sidebar options you have.
                          default_index=0) #Default side bar selection
    
    
# Sales Prediction Page
if (selected == 'Black Friday Sales Prediction'):
    
    # page title
    st.title('Black Friday Sales Prediction using ML')
    
    #Image
    st.image('hero.jpg')
    
    # getting the input data from the user
    col1, col2 = st.columns(2)
    
    with col1:
        Item_Visibility = st.number_input('Item Visibility', min_value=0.00, max_value=0.50, step=0.01)

    with col1:
        Item_MRP = st.number_input('Item MRP', min_value=30.00, max_value=300.00, step=1.00)

    with col1:
        Outlet_Size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])

    with col2:
        Discount_price = st.selectbox('Item Discount', ['20% discount', 'No discount'])

    with col2:
        Outlet_Location_Type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
    
    #Data Preprocessing
        
    data = {
            'Item_Visibility': Item_Visibility,
            'Item_MRP' : Item_MRP,
            'Outlet_Size' : Outlet_Size,
            'Discount_Price_Regular': Discount_Price,
            'Outlet_Location_Type' : Outlet_Location_Type
             }
    
    oe = OrdinalEncoder(categories = [['Small','Medium','High']])
    scaler = StandardScaler()
    
    def make_prediction(data):
        df = pd.DataFrame(data, index=[0])
        
        if df['Discount_Price_Regular'].values == '20% Discount':
            df['Discount_Price_Regular'] = 0.0
  
        if df['Discount_Price_Regular'].values == 'No Discount':
          df['Discount_Price_Regular'] = 1.0

        if df['Outlet_Location_Type'].values == 'Tier 1':
          df[['Outlet_Location_Type_Tier 1','Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3']] = [1.0, 0.0, 0.0]

        if df['Outlet_Location_Type'].values == 'Tier 2':
          df[['Outlet_Location_Type_Tier 1','Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3']] = [0.0, 1.0, 0.0]

        if df['Outlet_Location_Type'].values == 'Tier 3':
          df[['Outlet_Location_Type_Tier 1','Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3']] = [0.0, 0.0, 1.0]
        
        df['Outlet_Size'] = oe.fit_transform(df[['Outlet_Size']])
        df = df.drop(columns = ['Outlet_Location_Type'], axis = 1 )
        df[['Item_Visibility', 'Item_MRP']] = StandardScaler().fit_transform(df[['Item_Visibility', 'Item_MRP']])
        
        prediction = model.predict(df)
        
        return prediction
        

    
    # code for Prediction
    sales_prediction_output = ""
    
    # creating a button for Prediction
    
    if st.button('Predict Sales'):
        sales_prediction = make_prediction(data)
        sales_prediction_output = f"The sales is predicted to be {round(float(sales_prediction),2)}"

        
        st.success(sales_prediction_output)





    


