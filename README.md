# Handwritten-Digit-Recognition
The project uses the MNIST dataset, which consists of handwritten digits in a 28x28 pixel format.
## Installation
```bash
pip install numpy
pip install opencv-python
pip install matplotlib
pip install tensorflow
```

## Usage
Part 1
* Normalize the training and testing data
* Create and compile the model
* Train the model on the training data
* Save the model

Part 2
* Load the model
* Evaluate the model on the testing data
* Use the model to predict the digits in the "digits" folder
* You can add to the digits folder by drawing in Paint on a 28X28 pixels canvas
  
## Results

```python
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)
```
```bash
236.17965698242188
0.9642000198364258
```
```bash
This digit is probably a 0
```
![image](https://github.com/Covasan-Iosif/Handwritten-Digit-Recognition/assets/72391365/a8aeba89-49a5-419e-9f80-95a270e9d601)
```bash
This digit is probably a 2
```
![image](https://github.com/Covasan-Iosif/Handwritten-Digit-Recognition/assets/72391365/f1c4c61a-1131-46bf-bf49-640e7ea0366e)
