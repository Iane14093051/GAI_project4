
# How to use
1. Clone or download this repo.
2. Go to dip file. Placing some picture in "img-prior-in", and open "dip.py".
3. Replace the image name with your own picture. You can also change the "noise_intensity" ,"iterations" or "result[]" that you want to store or show.
4. Run "dip.py". After that you can get the result image in "output_images".
5. Go to ddpm file. Then you can start to train the ddpm model by run "Main.py". You also can modify the hyperparameter in this code.
6. Next, you need to place your input picture(dip) to the "SampledImgs" file, and modify the "image_path" in the "Diffusion/Train.py". 
7. Go to "Main.py" again and set the "state" to eval and "test_load_weight" to your training results which locate in "Checkpoints".
8. And now you can use Main.py to run the evaluation.
9. After the evaluation, you will get the final image in the "SampledImgs".
