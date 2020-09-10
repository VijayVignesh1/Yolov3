# YOLOv3 
<p align="center">
<img  src="images/yolov3.png">
</p>
<p align="center">
This is an implementation of YOLOv3 model trained on the Oxford Hands Dataset. <br> 
</p>

------------------------------
## Steps to train
1. Download the annotations and image dataset from [here](https://drive.google.com/file/d/1KHzFdt3ZpdOcvyGgmfdqZsn-8-088JO6/view?usp=sharing) and place it in the "data" folder. 
<p align="center">
<pre>
Yolov3
  |
  |
  |____data 
	 |
	 |
	 |____train 
		|
 	        |____ Buffy_0.jpg 
		|____ Buffy_0.txt 
			.
			.
			.
</pre>
</p>
2. If you already have a checkpoint file, include it in train.py. (checkpoint=checkpoint_file). <br>
3. Change the other model parameters, if necessary, in train.py file. <br>
4. Run the following command and watch the magic. :sunglasses: <br>

```.bash
python train.py
```

---------------------------------

## Steps to Predict
1. Change the testing image in predict.py. <br>
2. Include the desired checkpoint to be used in predict.py. <br>
3. Run the following command and watch the model magically detect the hands. :grinning: <br>

```.bash
python predict.py
```

----------------------------------

## Results
Below are the result of training the model on a small subset of Oxford Hands Dataset. <br>
### Graph of loss for 500 epochs (Epochs: 6500-7000)
<p align="center">
<img src="images/final-loss_7000.png">
</p>
<br>

### Validating the model on an image 
<br>
<p align="center">
<img src="images/Result.jpg">
</p>
<br>
With a little tweaking, the model can be trained on other datasets as well. :wink: <br> <br>

Go ahead, pull it, train it and have fun. :smiley: <br>

-------------------------------------
<p align="center">
:star: Happiness should be a function without any parameters :star: <br> <br>
<b>Happy Coding</b> :exclamation: :exclamation:
</p>
