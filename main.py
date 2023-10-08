import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px

from itertools import cycle
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import os
import plotly.graph_objects as go

import shutil

from flask import Flask, request, send_from_directory
from twilio.twiml.messaging_response import MessagingResponse


def predict():
    BTC_Ticker = yf.Ticker("BTC-USD")
    period = 30
    df = BTC_Ticker.history(interval = "1d", period=f"{period}d")
    df = df[['Close']]
    dataset = df.values
    dataset = dataset.astype('float32')
    scaler=MinMaxScaler(feature_range=(0,1))
    dataset=scaler.fit_transform(np.array(dataset).reshape(-1,1))


    if os.path.exists('images/'):
        shutil.rmtree('images/')
    os.mkdir('images/')

    model = load_model('weights-BTC.h5')
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

    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })

    names = cycle([f'Last {time_step} hours close price', f'Predicted next {pred_days} hours close price'])

    lstmdf=dataset.tolist()
    lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

    names = cycle(['Close price'])


    fig = px.line(x=list(range(-period, pred_days)), y=lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})
    fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

    fig.for_each_trace(lambda t: t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_image("images/pred.png")



# predict()

app = Flask(__name__)

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory('images',
                               filename, as_attachment=True)

@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():

    body = request.values.get('Body', None)

    response = MessagingResponse()

    if body == '!predict':
        predict()
        with response.message() as message:
            message.body = "{0}".format("Welcome to Mars.")
            message.media('https://rich-gobbler-hopefully.ngrok-free.app/uploads/{}'.format('pred.png'))

    return str(response)

if __name__ == "__main__":
    app.run(port=8000, debug=False)