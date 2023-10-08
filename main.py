import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px

import os
from twilio.rest import Client

from itertools import cycle
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import os
import plotly.graph_objects as go

import shutil

from flask import Flask, request, send_from_directory
from twilio.twiml.messaging_response import MessagingResponse


def predict():
    ticker = yf.Ticker("GAS-ETH")
    period = 30
    df = ticker.history(interval = "1d", period=f"{period}d")
    df = df[['Close']]
    dataset = df.values
    dataset = dataset.astype('float32')
    scaler=MinMaxScaler(feature_range=(0,1))
    dataset=scaler.fit_transform(np.array(dataset).reshape(-1,1))


    if os.path.exists('images/'):
        shutil.rmtree('images/')
    os.mkdir('images/')

    model = load_model('weights-GAS-ETH.h5')
    time_step = 15

    x_input=dataset[len(dataset)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=time_step

    i=0
    pred_days = 15
    while(i<pred_days):

        if(len(temp_input)>time_step):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]

            lst_output.extend(yhat.tolist())
            i=i+1

        else:

            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())

            lst_output.extend(yhat.tolist())
            i=i+1

    last_days=np.arange(1,time_step+1)

    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = scaler.inverse_transform(dataset[len(dataset)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    lstmdf=dataset.tolist()
    lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]


    fig = px.line(x=list(range(-period, pred_days)), y=lstmdf,labels={'value': 'Gas price','index': 'Timestamp'})
    fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(title_text='Plotting Ethereum Gas with prediction',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

    fig.write_image("images/pred.png")

    return np.array(lstmdf)

predict()

# app = Flask(__name__)

# account_sid = os.environ['TWILIO_ACCOUNT_SID']
# auth_token = os.environ['TWILIO_AUTH_TOKEN']
# client = Client(account_sid, auth_token)


# @app.route('/uploads/<path:filename>')
# def download_file(filename):
#     return send_from_directory('images',
#                                filename, as_attachment=True)

# @app.route("/sms", methods=['GET', 'POST'])
# def sms_reply():

#     body = request.values.get('Body', None)

#     if body == '!predict':
#         preds = predict()
#         message = client.messages \
#             .create(
#                  body=f'Predictions\n\n Best day for Algo to Eth: \n{np.argmax(preds)} days from now \n Conversion: {preds.max()} \n\nBest day for Eth to Algo: \n{np.argmin(preds)} days from now\n Conversion: {preds.min()}',
#                  from_='+18442608697',
#                  media_url='https://rich-gobbler-hopefully.ngrok-free.app/uploads/{}'.format('pred.png'),
#                  to='+18482189972'
#              )

# if __name__ == "__main__":
#     app.run(port=8000, debug=False)