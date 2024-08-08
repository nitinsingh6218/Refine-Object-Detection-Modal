from numpy import expand_dims
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
 
# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height
 
# load yolov3 model
model = load_model('Yolov3_224_custom_prll2_typr_512.h5')
# define the expected input shape for the model
input_w, input_h = 224,224
# define our new photo
photo_filename = '2.jpg'
photo_filename1 = '20.jpg'
photo_filename2 = '41.jpg'
photo_filename3 = 'images22.jpg'
photo_filename4 = 'images25.jpg'
# load and prepare image
image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
image1, image_w, image_h = load_image_pixels(photo_filename1, (input_w, input_h))

image2, image_w, image_h = load_image_pixels(photo_filename2, (input_w, input_h))

image3, image_w, image_h = load_image_pixels(photo_filename3, (input_w, input_h))

image4, image_w, image_h = load_image_pixels(photo_filename4, (input_w, input_h))

# make prediction
yhat = model.predict(image)
yhat1 = model.predict(image1)
yhat2 = model.predict(image2)
yhat3 = model.predict(image3)
yhat4 = model.predict(image4)

print(yhat)
print(yhat.argmax())
print(yhat1.argmax())
print(yhat2.argmax())
print(yhat3.argmax())
print(yhat4.argmax())
sc=np.argmax(yhat, axis=-1)[0]
sc1=np.argmax(yhat1, axis=-1)[0]
sc2=np.argmax(yhat2, axis=-1)[0]
sc3=np.argmax(yhat3, axis=-1)[0]
sc4=np.argmax(yhat4, axis=-1)[0]
'''if sc==0:
    print("Ambulance")
elif sc==1:
    print("Armor_vehicle")
elif sc==2:
    print("Army_trucks")
elif sc==3:
    print("Containers_truck")
elif sc==4:
    print("Fire_fighter_vehicle")
elif sc==5:
    print("Government_Official_vehicle")
elif sc==6:
    print("JCB")
elif sc==7:
    print("Police_Car")
elif sc==8:
    print("Police_bike")
elif sc==9:
    print("Road_roller")
else:
    print("Regular vehicle")'''
dict={0:'Ambulance',  1:'Armor_vehicle',2:'Army_trucks',
 3:'Containers_truck',4:'Fire_fighter_vehicle',5:
 'Government_Official_vehicle',6:
 'JCB', 7:'Police_Car', 8:'Police_bike',9:'Road_roller'}
print(dict[sc])
print(dict[sc1])
print(dict[sc2])
print(dict[sc3])
print(dict[sc4])
# summarize the shape of the list of arrays
print([a.shape for a in yhat])
