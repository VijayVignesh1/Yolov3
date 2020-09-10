# YOLOv3 
This is an implementation of YOLOv3 model trained on the Oxford Hands Dataset.
# Steps to train
1. Download the annotations and image dataset from [here] :https://drive.google.com/file/d/1KHzFdt3ZpdOcvyGgmfdqZsn-8-088JO6/view?usp=sharing and place it in the "data" folder. <br>
	data <br>
	 |<br>
	 |<br>
	 |_\__\_\_train <br>
		 |<br>
		 |_\_\_\_ Buffy_0.jpg <br>
		 |_\_\_\_ Buffy_0.txt <br>
			....
2. If you already have a checkpoint file, include it at train.py. (checkpoint=checkpoint_file) <br>
3. Change the other model parameters, if necessary, at train.py file. <br>
4. Execute "train.py" and watch the magic. B-) <br>

# Steps to Predict
1. Change the testing image at predict.py. <br>
2. Include the desired checkpoint to be used at predict.py. <br>
3. Execute "predict.py" and watch the model magically detect the hands. :D <br>

Below are the result of training the model on a small subset of Oxford Hands Dataset. <br>
1. Graph of loss for 500 epochs (Epochs: 6500-7000)
![](images/final-loss_7000.png)<br><br>
2. Validating the model on an image <br><br>
&nbsp; &nbsp; &nbsp; ![](images/Result.jpg)<br><br>

With a little tweaking, the model can be trained on other datasets as well. ;) <br>

Go ahead, pull it, train it and have fun. :) <br>





