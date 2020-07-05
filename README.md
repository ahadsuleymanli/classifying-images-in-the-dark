### Create the envs:
- start your preffered virtual environment in project root
- pip install -r requirements.txt
- cd node_add && npm init

### How to run:
- run python inference_api.py in one shell
- and then run npm start --prefix ./node_app in another shell
- go to localhost:3000 in your browser to try the app out
- images have to be jpeg variations for now


### about the models:
- dnn2: inputs are 28x28 image thats is passed into 2x Conv layers and a (64,) shape HSV intensity histohram that is passed into a Dense layer.
- dnn: same as dnn2 but only takes the image as input
- svm: takes a (64,) shape HSV intensity histohram as input

### accuracies:
- dnn2: %84
- dnn: %81
- svm: %75