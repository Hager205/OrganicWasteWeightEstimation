import warnings
warnings.filterwarnings("ignore")
from io import BytesIO
from PIL import Image
from inference_sdk import InferenceHTTPClient
from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS, MAX_ACTIVE_MODELS
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference.core.workflows.execution_engine.core import ExecutionEngine
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
# import cv2
# import requests as req
import base64
import re
import json
import shutil
import os
from tqdm import tqdm
import requests

materials = json.load(open('material_properties.json'))

def decode_base64_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))


def predict(img_path, is_url=False):

    image = img_path
    if is_url:
        response = requests.get(img_path)
        image_bytes = response.content
        image = image_bytes
        

    seg_client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="Sn1vSUk86V5QpQ25WfdA"
    )

    seg_result = seg_client.run_workflow(
        workspace_name="taco-9911u",
        workflow_id="custom-workflow",
        images={"image": img_path},
        use_cache=True
    )


    all_predictions_path="all_predictions"
    os.makedirs(all_predictions_path, exist_ok=True) 

    current_predictions_path="current_predictions"
    if os.path.exists(current_predictions_path):
        shutil.rmtree(current_predictions_path)
    os.makedirs(current_predictions_path)

    
    for i, img in tqdm(enumerate(seg_result[0]['dynamic_crop']),desc="Extracting Objects"):
        class_name = seg_result[0]['model_predictions']['predictions']['predictions'][i].get('class').lower()
        confidence = round(seg_result[0]['model_predictions']['predictions']['predictions'][i].get('confidence'), 2)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        image = decode_base64_image(img)
        plt.imsave(f"{current_predictions_path}/{timestamp}_{confidence}_{class_name}.png", np.array(image))
        plt.imsave(f"{all_predictions_path}/{timestamp}_{confidence}_{class_name}.png", np.array(image))


    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = ModelManager(model_registry=model_registry)
    model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager
    }

    results=dict()

    label_visualization_img = decode_base64_image(seg_result[0]['label_visualization'])
    plt.imsave(f"{current_predictions_path}/label_visualization.png", np.array(label_visualization_img))


    for img_name in tqdm(os.listdir(current_predictions_path),desc="Classifying Objects"):

        if img_name.endswith('.png') and img_name != 'label_visualization.png':
            class_name=re.search(r'_\d+\.\d+_(.*?)\.png$', img_name).group(1)
            class_name_temp=class_name
            img_id=re.search(r'(20[^_/]*)_', img_name).group(1)
            seg_condifence=float(re.search(r'_(\d+\.\d+)_', img_name).group(1))
            food_classification_confidence = None


            if class_name == 'food waste':
                food_class_client = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key="0YHBpNOGQLWKMICOfU27"
                )

                food_class_result = food_class_client.run_workflow(
                    workspace_name="foodclass-mitcr",
                    workflow_id="custom-workflow",
                    images={
                        "image": f"{current_predictions_path}/{img_name}"
                    },
                    use_cache=True # cache workflow definition for 15 minutes
                )
                class_name = json.loads(food_class_result[0]['google_gemini']['output'])['class_name']
                food_classification_confidence = json.loads(food_class_result[0]['google_gemini']['output'])['confidence']


            class_placeholder = class_name.replace("other ", "")

            CUSTOM_WORKFLOW = {
                "version": "1.0",
                "inputs": [
                    {"type": "WorkflowImage", "name": "image"}
                ],
                "steps": [
                    {
                        "type": "roboflow_core/google_gemini@v1",
                        "name": "google_gemini",
                        "images": "$inputs.image",
                        "model_version": "gemini-2.0-flash-exp",
                        "temperature": 0,
                        "api_key": "AIzaSyD5UvtCm4ccw2w5cBV7UJ-wz9RCfPPv1Gs",
                        "task_type": "structured-answering",
                        "output_structure": {
                            "object_description": f"Description for the {class_placeholder} shown in the image",
                            "estimated_dimensions_cm": f"Estimated dimensions for the {class_placeholder} shown in the image (in centimeters)",
                            "shape_assumption": f"Shape assumption for the {class_placeholder} shown in the image",
                            "volume_estimate_ml": f"Volume estimation for the {class_placeholder} shown in the image (a single number in milliliters)",
                            "assumptions": "Estimated dimensions based on the nearby objects for scale",
                            "confidence_score": "Confidence score"  
                        }
                    }
                ],
                "outputs": [
                    {"type": "JsonField", "name": "google_gemini", "selector": "$steps.google_gemini.*"}
                ],
            }


            execution_engine = ExecutionEngine.init(
                workflow_definition=CUSTOM_WORKFLOW,
                init_parameters=workflow_init_parameters,
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            )


            result = execution_engine.run(
                runtime_parameters={
                    "image": [f"{current_predictions_path}/{img_name}"]
                }
            )

            output = json.loads(result[0]['google_gemini']['output'])
            pre_volume_estimate_ml = str(output['volume_estimate_ml'])
            numbers = re.findall(r'\d+\.?\d*', pre_volume_estimate_ml)
            if numbers:
                float_numbers = np.array(numbers).astype(float)
                output['volume_estimate_ml'] = float(float_numbers.mean())
            else: # If the LLM couldn't estimate
                volume_averages = {
                    'plastic straw': 3.93,
                    'clear plastic bottle': 1000,
                    'corrugated carton': 60200,
                    'crisp packet': 1764,
                    'disposable food container': 1000,
                    'disposable plastic cup': 475,
                    'egg carton': 2100,
                    'foam cup': 414,
                    'magazine paper': 3.015,
                    'meal carton': 2070,
                    'other plastic bottle': 750,
                    'other plastic wrapper': 0.18,
                    'plastic bottle cap': 6.77,
                    'plastic film': 50,
                    'plastic glooves': 1.2,
                    'plastic lid': 15.3,
                    'plastic utensils': 24.8,
                    'polypropylene bag': 15000,
                    'single-use carrier bag': 23625,
                    'styrofoam piece': 750,
                    'food waste': 664
                }
                output['volume_estimate_ml'] = volume_averages[class_name_temp]


            results[img_id] = {
                "volume_ml": output['volume_estimate_ml'],
                "class": class_name,
                "segmentation_confidence": seg_condifence,
                "volume_est_confidence": output['confidence_score']
            }

            if class_name_temp=='food waste':
                results[img_id]['food_classification_confidence'] = food_classification_confidence

            results[img_id]['density'] = materials[class_name]['density']
            results[img_id]['hydrogen_ratio'] = materials[class_name]['hydrogen_ratio'] 

    return {
        'segmentation_model_results': seg_result,
        'objects_info': results    
        }

print('h_extraction_pipeline loaded')       