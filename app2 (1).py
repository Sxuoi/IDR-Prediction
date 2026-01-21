import streamlit as st
import numpy as np
import pandas as pd
import pickle
from numpy import array
from datetime import datetime 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load the LSTM model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Load the model
model_path = 'C:\\Users\\stafg\\Desktop\\.ipynb_checkpoints\\currency_prediction_model.pkl'  # Replace with the path to your saved model file
try:
    model = load_model(model_path)
except FileNotFoundError:
    st.error("Error: Model file does not exist at the specified path.")

maindf = pd.read_csv("https://raw.githubusercontent.com/Sxuoi/UNI_Assign/main/IDRtoUSD.csv")
closedf = maindf[['Date','Close']]
closedf = closedf[closedf['Date'] > '2021-05-23']
close_stock = closedf.copy()

del closedf['Date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

training_size=int(len(closedf)*0.60)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

# Function to preprocess data
def date_procesing(selected_date):
    # Assuming closedf is your DataFrame containing the date and close price
    year = selected_date.year
    month = selected_date.month
    day = selected_date.day

    # Membuat objek timestamp berdasarkan input bulan dan tahun
    selected_date = pd.Timestamp(year=year, month=month, day=day)
    latest_date = pd.Timestamp(year=2024, month=5, day=24)
    days_input = (selected_date - latest_date ).days
    st.write(days_input)
    return days_input

# Function to make predictions
def make_predictions(date_procesing, model):
    
    time_step = 15
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    

    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = date_procesing
    while(i<pred_days):
    
        if(len(temp_input)>time_step):
        
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
        
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
       
            lst_output.extend(yhat.tolist())
            i=i+1
        
        else:
        
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
        
            lst_output.extend(yhat.tolist())
            i=i+1
               
    st.write("Output of predicted next days: ", len(lst_output))
    
    last_days=np.arange(1,time_step+1)
    
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })

    return new_pred_plot

# Main Streamlit app
def main():
    st.title("Prediksi Harga Dolar")
    html_temp = """<div style = "background-color:purple;padding:7px"> 
    <h2 style = "color:black;text-align:left"> Prediksi berdasarkan fluktuasi harga pasar</h2>
    </div>"""   
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input for date
    # Input hari, bulan dan tahun
    selected_date = st.date_input("Pilih tanggal:", value=datetime.today())
    
    # Display the predicted data
    #predicted_data = pd.DataFrame({'Date': [selected_date], 'Predicted_Close': predictions.flatten()})
    #st.write("Predicted Data:")
    #st.write(predicted_data)
    if st.button ("Prediksi"):
        
        time_step = 15
        x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
    
        lst_output=[]
        n_steps=time_step
        i=0
        pred_days = int(date_procesing(selected_date))
        while(i < pred_days):
    
            if(len(temp_input)>time_step):
        
                x_input=np.array(temp_input[1:])
                #print("{} day input {}".format(i,x_input))
                x_input = x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
        
                yhat = model.predict(x_input, verbose=0)
                #print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
       
                lst_output.extend(yhat.tolist())
                i=i+1
        
            else:
        
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
        
                lst_output.extend(yhat.tolist())
                i=i+1
               
        #st.write("Output of predicted next days: ", len(lst_output))
    
        last_days=np.arange(1,time_step+1)
    
        temp_mat = np.empty((len(last_days)+pred_days+1,1))
        temp_mat[:] = np.nan
        temp_mat = temp_mat.reshape(1,-1).tolist()[0]

        last_original_days_value = temp_mat
        next_predicted_days_value = temp_mat

        last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
        next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

        new_pred_plot = pd.DataFrame({
            'last_original_days_value':last_original_days_value,
            'next_predicted_days_value':next_predicted_days_value
        })
    
        names = cycle(['Close price 15 hari terakhir', f'Prediksi close price {pred_days} hari kedepan'])

        # Create the Plotly figure
        fig = px.line(new_pred_plot, x=new_pred_plot.index,
              y=[new_pred_plot['last_original_days_value'], new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Harga Dolar', 'index': 'Timestamp'})

        # Update layout
        fig.update_layout(title_text=f'Perbandingan 15 hari terakhir dengan {pred_days} hari kedepan',
                    plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

        # Update trace names using the names cycle
        fig.for_each_trace(lambda t: t.update(name=next(names)))

        # Update axes visibility
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Display the Plotly figure using Streamlit
        st.plotly_chart(fig)
        

if __name__ == "__main__":
    main()
