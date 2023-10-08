import os
import json
import uuid
import base64
import shutil
import requests
import http.client

import numpy as np
import yfinance as yf
import plotly.express as px

from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256
from Crypto.Cipher import PKCS1_OAEP

from twilio.rest import Client
from keras.models import load_model
from flask import Flask, request, send_from_directory
from sklearn.preprocessing import MinMaxScaler

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
    fig.update_layout(title_text='Plotting Ethereum Gas with Prediction',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock',
                      xaxis_title="Days from Today", yaxis_title="Gas")

    fig.write_image("images/pred.png")

    return np.array(lstmdf)

def createNewCipherText():
    entity_secret = bytes.fromhex(os.environ['ENTITY_SECRET'])

    if len(entity_secret) != 32:
        print("invalid entity secret")
        exit(1)

    public_key = RSA.importKey("-----BEGIN PUBLIC KEY-----\n\
        MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAyVqtPHWiGMJEIDnnGhV8\n\
        X8nvuw5DlUFUNhzHCkDtmNN/emSO73c6oumg6VUhJlWZPB0qw9taYlecM0PEue3m\n\
        F0nEZBptQ54m1gLOwabtZ6pfqt42GaOjpVZ7g+BjcmEcO71rkQ8t3PPFiKU1zFgR\n\
        UAecEc60qEkpmuJynU/hMtwaHFqThJ5fr5Cd6A3/BNmND2sHgNaKYccqDM9ecm+9\n\
        Dgs76Wgyo3f4H21It3sNKlwYj0Ro0pqbzh8qtC09F4YRJEUna0NWQvAOYQzIjzzc\n\
        IMg48KnJNsMgIoxVNFYSh9yPRUeVgnFtBATRhbsPf+vmaYzivZmE/6PlbWMHnfOR\n\
        YSMYwsaUKOGSPdQE4ScccdetAU2VPRzPnEtbzPQ3TgmqoNZF6ZhlWbaHyud+RtV6\n\
        jzfYitOe2SO7TofwBhuR0T9orxO+opFgZP9FNXjHFWmKJ5KKEZMx9LwwrVnKnFq0\n\
        NGDUvsw6VOL0k/b8pG9X+ubC5bhSnKeyHuBgSiq3gGFWakLOJghjTA7A7capdoLf\n\
        AgwPMRvPi0mZ9+RoUg2JfJzzLiL+AJMsTqme37suhJYjbkmq8CwMmupURTnE6AUJ\n\
        4XH7TqcnMp0vUkyngkclcIWBbK+O4TZnaRN1WOA8sna8Wjy5HRzTO4P+ykZ5bViQ\n\
        ZbpdAkqwIzAkje5dqFcvLCsCAwEAAQ==\n\
        -----END PUBLIC KEY-----"
    )

    cipher_rsa = PKCS1_OAEP.new(key=public_key, hashAlgo=SHA256)
    encrypted_data = cipher_rsa.encrypt(entity_secret)
    encrypted_data_base64 = base64.b64encode(encrypted_data)
    return encrypted_data_base64.decode()

app = Flask(__name__)

account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)


@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory('images',
                               filename, as_attachment=True)

@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():

    body = request.values.get('Body', None)

    if body == '!predict':
        preds = predict()
        message = client.messages \
            .create(
                 body=f'Predictions\n\n Best day for ethereum gas: \n{np.argmin(preds)} days from now \n Estimated Rate: {preds.min()}',
                 from_='+18442608697',
                 media_url='https://rich-gobbler-hopefully.ngrok-free.app/uploads/{}'.format('pred.png'),
                 to='+18482189972'
             )

    elif body == '!balance':
        conn = http.client.HTTPSConnection("api.circle.com")

        headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer {os.environ['CIRCLE_API_KEY']}"
        }

        conn.request("GET", f"/v1/w3s/wallets/{os.environ['WALLET_1_ID']}/balances", headers=headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode('utf-8'))

        balanceString = None
        for token in data['data']['tokenBalances']:
            tokenID = token['token']['id']
            amount= token['amount']
            unit = token['token']['name']

            tokenString = f"Token ID: {tokenID}\nAmount: {amount} {unit}"
            if balanceString is None:
                balanceString = tokenString
            else:
                balanceString += f"\n\n{tokenString}"

        message = client.messages \
            .create(
                 body=balanceString,
                 from_='+18442608697',
                 to='+18482189972'
             )


    elif body == '!transfer':
        url = "https://api.circle.com/v1/w3s/transactions/transfer/estimateFee"

        payload = {
            "amounts": ["10"],
            "destinationAddress": os.environ['WALLET_2_ADDRESS'],
            "sourceAddress": os.environ['WALLET_1_ADDRESS'],
            "tokenId": os.environ['SAMPLE_ETH_TOKEN_ID']
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {os.environ['CIRCLE_API_KEY']}"
        }

        response = requests.post(url, json=payload, headers=headers)

        parsed_data = response.json()
        low_max_fee = parsed_data['data']['low']['maxFee']
        medium_max_fee = parsed_data['data']['medium']['maxFee']
        high_max_fee = parsed_data['data']['high']['maxFee']

        # Print the maxFee values
        print("Low Max Fee:", low_max_fee)
        print("Medium Max Fee:", medium_max_fee)
        print("High Max Fee:", high_max_fee)
        message = client.messages \
            .create(
                 body=f'Transfer 1 USDC\n\nEstimated Gas:\nLow Max Fee:{low_max_fee}\nLow Max Fee:{medium_max_fee}\nLow Max Fee:{high_max_fee}\n\nConfirm?',
                 from_='+18442608697',
                 to='+18482189972'
             )

    elif body=='yes':

        url = "https://api.circle.com/v1/w3s/developer/transactions/transfer"

        payload = {
            "amounts": ["1"],
            "idempotencyKey": str(uuid.uuid4()),
            "destinationAddress": os.environ['WALLET_2_ADDRESS'],
            "entitySecretCiphertext": createNewCipherText(),
            "tokenId": os.environ['SAMPLE_ETH_TOKEN_ID'],
            "walletId": os.environ['WALLET_1_ID'],
            "feeLevel": "LOW"
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "Bearer TEST_API_KEY:d237343fb44c9582c0280e5f43297d8d:017d2bf54597051de1d4e1439a44f6b1"
        }

        response = requests.post(url, json=payload, headers=headers)

        message = client.messages \
            .create(
                 body=f"Transfer Initiated\n{response.json()['data']['id']}",
                 from_='+18442608697',
                 to='+18482189972'
             )

if __name__ == "__main__":
    app.run(port=8000, debug=False)