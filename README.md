Task general description In this task you will Build a flower classifier with task transfer from a pre-trained network (Alex Net). Try to Improve the classifier using at least two of several options: data augmentation in training, weight decay, using a better network, or other ideas. Test your Accuracy and report your results The exercise goals: experience building a classification pipe with CNN components. Get acquainted with some possible methods for pipe improvement.

Datasets: The flowers dataset was collected in Volkani institute: The data includes 473 cropped images of flowers and non-flowers, with corresponding labels. The original images from which the flower and non-flower rectangles were cropped from are not given, due to their large size. The data is in 'FlowerData.zip'. It includes the images and the labels 'FlowersDataLabels.mat'. Use the command scipy.io.loadmat() to read it. Use the first 300 cropped images for training. Test on the remaining 173. You will have to resize the images to 2242243, which is the input shape of the pre-trained CNNs on imagenet. Notice that in some network implementations the input shape might be 2272273.

Task description and options: The Basic pipe includes: Initiate an Alexnet -based architecture (using the keras public library): Alexnet implementation was built to classify images on Imagenet datset, which has 1000 classes. Hence the output layer has 1000 neurons. An example to code initializing the architucture and running the model can found here. The architecture is

In this project, the classification is flower or not flower. Thus, the output layer should have a single neuron estimating p ̂(y=flower|x) (i.e. the probability that an image is a flower). We will drop the last layer of Alex-Net. Instead, denote the before-last of Alex net by X_bl∈R^4096. The output neuron p for our net will be obtained as

p=σ(w_L⋅X_bl-b_L)

The loss: Assume an example x with the label y (y=1 means that the image x is a flower, y=0 means it is not), and the network output is p(x)=p ̂(y=1|x). We minimize 
loss(y,p(x))={■(log⁡p&y=1@log⁡〖(1-p)〗&y=0)

For a single image. The total loss is L=∑_(i=1)^N▒〖loss(〗 y_i,p(x_i)) This network loss in keras is called "binary_crossentropy".

 Important remark – Keras runs on top of other deep learning frameworks. In this work, use Tensorflow as keras backend.

Task transfer: use pre-trained Alexnet weights (trained on imagenet dataset) as the initial weights of the network (They can be downloaded in this link . They can be loaded to the network using a single command model.load_weights(‘my_weights’) of keras or as in this example ).
Fine tune the network weights for the specific problem. Tuning all the network weights is not necessarily the best idea (error-wise and time-wise), it is recommended to start by trying the tuning the final layers only.
In addition to implementing the basic CNN pipe, you should try at least two ways to improve its results. Options are (Individuals may try only one): Data Augmentation: Add to the training data (but not to the test data) examples created by applying horizontal flip and/or mild cropping of the original training images. A 3D image can be horizontally flipped by taking I2=I[:, ::-1, :]. When cropping, use random crops of large portions (>80-90%) of the original image. More explanations and examples can be found here. Weight decay (and tuning of it) Dropout (and tuning of it): a very commonly used regularization technique. Additional information here. Fine tune by re-training all of AlexNet layers to the task (note that such full re-training may take many hours). Neural decay (and tuning of it) Using a more complex network: Instead of Alex net, you may try to use more advanced networks like VGG-16, inception-v3 or Resnet. Note that more advanced network may be more accurate, but are usually considerably slower than AlexNet. You are welcome to purpose other ideas and report their results.
