# Course Synopsis

This course provides an overview of the fundamental theories behind generative AI, covering various generative models for both text and images. It also includes practical exercises for applying these concepts in real-world scenarios.

# Objective
-The motivation behind this approach is to accelerate the slow backward learning process in DDPM by utilizing the image-specific priors automatically learned by the CNN architecture in DIP.

# Usage

## 1. Clone or Download the Repository
- Clone this repository or download it as a ZIP file.

## 2. Prepare the DIP Files
- Navigate to the `dip` folder.
- Place your images in the `img-prior-in` directory.
- Open `dip.py` and replace the image name with your own picture. You can also adjust the `noise_intensity`, `iterations`, or `result[]` parameters as desired.

## 3. Run the DIP Script
- Execute `dip.py`. The resulting images will be saved in the `output_images` folder.

## 4. Prepare for DDPM Training
- Navigate to the `ddpm` folder.
- Start training the DDPM model by running `Main.py`. You can modify the hyperparameters in this script as needed.

## 5. Set Up for Evaluation
- Place your input picture (from the DIP process) in the `SampledImgs` directory.
- Modify the `image_path` variable in `Diffusion/Train.py` to point to your input image.

## 6. Run Evaluation
- Return to `Main.py` and set the `state` to `eval`.
- Specify the `test_load_weight` to point to your training results located in the `Checkpoints` folder.
- Now, you can run `Main.py` to execute the evaluation.

## 7. View Results
- After the evaluation completes, the final images will be saved in the `SampledImgs` folder.

## **Note**
- Please refer to the [PDF file](https://github.com/Iane14093051/GAI_project4/blob/main/E14093051_GAI_Project4.pdf) for more detailed information about the experiment results.
- Please refer to the [PDF file](https://github.com/Iane14093051/GAI_project4/blob/main/GenAI_assignment_visual_signal.pdf) for more detailed information regarding the project requirements.
