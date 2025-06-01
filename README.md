# Crop Disease Detection App 🌿🦠

A deep learning-powered web app built using **Streamlit** and **TensorFlow** to identify crop diseases from leaf images. This project uses the PlantVillage dataset to classify images into 16 different crop disease categories.

---

## 📂 Project Structure
crop_disease_detection_app/
├── model/
│   └── crop_disease_model.h5         # Trained Keras model
├── streamlit_app.py                  # Streamlit web app
├── train_model.py                    # Model training script
├── utils/
│   └── downsample.py                 # Optional downsampling script
├── requirements.txt                  # Dependencies
├── README.md                         # Project 

---

## ⚙️ Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vijaykonidana28/crop_disease_detection_app.git
   cd crop_disease_detection_app
2.	Create a virtual environment:
   python3 -m venv tfenv
   source tfenv/bin/activate
3.	Install the dependencies:
   pip install -r requirements.txt
4.	Place the dataset:
	•	Download the PlantVillage dataset from:
https://github.com/spMohanty/PlantVillage-Dataset
	•	Extract it and place the folder in the root directory of this project.
	•	Rename the folder to PlantVillage.

   🚀 Run the App

To launch the Streamlit web app:
streamlit run streamlit_app.py

To retrain the model:
python train_model.py

🧠 Model Details
	•	Input image size: 150x150
	•	Architecture: Simple CNN with 2 Conv2D layers + Dense layers
	•	Number of classes: 16
	•	Format: Keras .h5

⸻

❌ Note
	•	The dataset (PlantVillage/) is excluded from this repository due to its large size.
 
⸻
👤 Author

Vijay Kumar Konidana
____

📄 License

This project is licensed under the MIT License.
