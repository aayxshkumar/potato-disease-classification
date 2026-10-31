# Potato-disease-classification

![CI](https://github.com/aayxshkumar/potato-disease-classification/actions/workflows/ci.yml/badge.svg)

A convolutional neural network that classifies potato leaf images as early blight, late blight or healthy, plus a small FastAPI prediction endpoint and a React upload page.

I built it by following the codebasics project [codebasics/potato-disease-classification](https://github.com/codebasics/potato-disease-classification). `Training/potato-disease-classification-model.ipynb` is that project's notebook, and `api/main.py` is adapted from its API.

## Contents

All paths are under `potato-disease-classification/`.

- `Training/PlantVillage/`: 2,152 leaf images in three class folders (1,000 early blight, 1,000 late blight, 152 healthy) from the Kaggle [PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village). `Training/dataset/` holds the same images split into `train`, `val` and `test` folders.
- `Training/model-training.ipynb`: loads `PlantVillage/` with `image_dataset_from_directory` (256x256 images, 80/10/10 split), trains the CNN for 50 epochs and saves numbered models to `../models/`.
- `Training/potato-disease-classification-model.ipynb`: the codebasics notebook for the same pipeline, with explanations and embedded videos.
- `Training/Untitled.ipynb`: an unfinished `ImageDataGenerator` experiment on `dataset/`. Its training cell failed in the saved output because scipy was not installed.
- `models/1`, `models/2`, `models/3`: trained models in TensorFlow SavedModel format.
- `api/main.py`: FastAPI app that loads `../models/1` and serves `GET /ping` and `POST /predict` (multipart `file`), returning the predicted class and confidence on `localhost:8000`.
- `frontend/`: React (Create React App, Material UI) page that posts an image to `REACT_APP_API_URL` and shows the result.

## Requirements

- Notebooks: Python 3.12 (the version CI uses), Jupyter and the libraries in `requirements.txt`.
- API: `api/requirements.txt.txt` pins TensorFlow 2.5.0, which needs Python 3.9 or older.
- Frontend: Node.js and npm.

## Setup and usage

```bash
git clone https://github.com/aayxshkumar/potato-disease-classification.git
cd potato-disease-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open the notebooks from `potato-disease-classification/Training/`; they read the image folders and write models with paths relative to that folder.

API, in a Python 3.9 or older environment, run from `api/` because the model path is relative:

```bash
cd potato-disease-classification/api
pip install -r requirements.txt.txt
python main.py
```

Frontend: copy `frontend/.env.example` to `frontend/.env` and set `REACT_APP_API_URL` to the predict endpoint (for example `http://localhost:8000/predict`), then:

```bash
cd potato-disease-classification/frontend
npm install
npm run start
```

The API allows CORS requests from `http://localhost` and `http://localhost:3000`.

## Running it today

The notebooks have not been re-run with the pinned versions. With TensorFlow 2.21 and Keras 3:

- `layers.experimental.preprocessing.*` (`Resizing`, `Rescaling`, `RandomFlip`, `RandomRotation`) no longer exists; the same layers are available as `layers.Resizing` and so on.
- `model.save("../models/N")` raises `ValueError`; Keras 3 needs a `.keras` or `.h5` file name.
- `tf.keras.models.load_model("../models/1")`, as used by the API, cannot load SavedModel directories. Keras 3 suggests `keras.layers.TFSMLayer` for inference-only loading.

## CI

CI validates the notebook JSON with nbformat, checks that the notebooks' libraries import, and byte-compiles `api/main.py`. It does not run the notebooks, the API or the frontend.
