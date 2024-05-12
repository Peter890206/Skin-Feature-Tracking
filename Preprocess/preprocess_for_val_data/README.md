# Preprocess for validation data

## Summary 
This folder is for generating the validation data and preprocess the data.
The **validation_data_create.py** is to select the frames in our experimental video, and save them in a folder.
After that, we use the Remove.bg on the website to remove the background (become white) for the frames we selected.
The **data_preprocess.py** is to crop the images in the folder to 32*32 with a stride of 31, which means each small window does not overlap.
Subsequently, **data_preprocess.py** also will remove the winodws whose white ratio is more than 0.45.

We use these two code to generate the test (validation) data to evaluate our DFE model.

[added by Peter]

## How to use **validation_data_create.py**?
**1.** Firstly, you need to change the path of the input video, and the path of the output folder in function 'crop_and_save_image'.

**2.** Setup the size of the output images.

**3.** Run the code.

**4.** Use the bar upon the video to select the approximate frame.

**5.** Use "i", "k", "j", "l" slightly to change the frame, "j" and "l" is for moving forward and backward 1 frame.
The "i" and "k" is for moving forward and backward a interval frame, where the interval is set by you.
When I was generating the val data, I set the interval to 80 (1/3 second).

**6.** After you select the targer frame, use "w", "s", "a", "d" to move the marker which is the center of the cropped image with a stride of 1, and you can use "z" to change the stride from 1 to 10 or 10 to 1.

**7.** Use "enter" to select the center point, at this time you can check the output image in your output folder.

**8.** If you want to change the position of the center, you can just do the step **6. to 7.** again.
The newest position in the same frame will cover the old one, so you can check the difference in your output folder.

**9.** After you make sure the output image is correct, repeat the step **5. to 7.**, you also can do the step **4.** to change the motion.


## How to use **data_preprocess.py**?
**1.** Go down to the bottum of the code, and change the path of the input folder and the path of the output folder.

**2.** If you want use the other crop size, you can change here.

**3.** If you want to change the parameters for remove the data with large white ratio, you can change in the function 'process_image'.

**4.** If you hardware cannot support the multi-Process, you can uncomment the for-loop code in the bottum.

**5.** Run the code.









