# Smart Solutions for Green Hydrogen Production

![Project Overview]([LINK_TO_PROJECT_IMAGE](https://github.com/Hager205/OrganicWasteWeightEstimation/blob/main/Results/Visuals/current_predictions/project_idea.png))

## Project Overview
This project focuses on classifying organic and non-organic waste using computer vision and deep learning. The goal is to accurately identify organic materials, which can then be utilized for sustainable green hydrogen production. By separating organic waste from non-organic, this system contributes to cleaner energy generation and effective waste management.

The system detects objects like bottles, cartons, and food waste, estimates their volume using 3D modeling techniques, and calculates their weight using density values. These accurate weight estimations are then used for sustainable green hydrogen production.

## Tools & Technologies
- Python – Main programming language
- Jupyter Notebook – For experimentation and workflow
- YOLOv8 – Object detection and classification
- OpenCV – Image processing
- NumPy & Pandas – Data manipulation
- Matplotlib – Visualization of results
- JSON – Storing input material properties and classification results

## Project Structure

- `Documentation Files/`  
  - `GRADUATION PROJECT DOCUMENTATION.pdf` – The full project report and documentation  

- `results/`  
  - `visual_results/` – Images showing current predictions  
  - `results_with_summary_info.json` – JSON file containing summary of project results  

- `data/`  
  - `material_properties.json` – Input properties of waste materials  

- `Technical Segmentation Model/`  
  - `Taco-Segmentation-Model.ipynb` – Notebook containing full code and workflow
  - `pipeline.py` – Pipeline script to run the complete project workflow  

- `requirements.txt` – Python libraries required to run the project

## How to Use
1. Download or clone the repository.
2. Open the notebook or run the pipeline script to process the data.
3. Check classification results in `visuals/classification_results/`.

## Results Examples

### Example 1: Bottle Classification
- **Input:** Image of a bottle  
- **Output Result:** Classification mask, segmented bottle, volume & weight estimation  
- ![Bottle Example](LINK_TO_BOTTLE_RESULT_IMAGE)

### Example 2: Carton Classification
- **Input:** Image of a carton  
- **Output Result:** Classification mask, segmented carton, volume & weight estimation  
- ![Carton Example](LINK_TO_CARTON_RESULT_IMAGE)

### Example 3: Food Waste Classification
- **Input:** Image of food waste  
- **Output Result:** Segmentation and weight estimation  
- ![Food Waste Example](LINK_TO_FOOD_RESULT_IMAGE)

> **Note:** Replace the `LINK_TO_...` with the actual GitHub URL of the uploaded images.

## Authors
Hager Mohamed – Junior Data Scientist & Software Engineer
