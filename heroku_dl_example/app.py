from flask import Flask

from keras.models import load_model
import numpy as np

app = Flask(__name__)

@app.route('/dl_example')
def example():
    # 載入模型
    print('載入模型')
    model_load = load_model('model.h5')

    # 讀取input
    print('讀取input')
    test = np.load('x_test.npy')

    # input前處理
    print('input前處理')
    test = test / 255
    test = test.reshape(1, 28, 28, 1)

    print('測試input')
    pred = model_load.predict_classes(test)

    return 'predict: ' + str(pred[0])

#  Application 運行（本機版）
# if __name__ == "__main__":
#     app.run(host='0.0.0.0')

#  Application 運行（heroku版）
import os
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=os.environ['PORT'])




