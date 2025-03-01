import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import csv
#from st_aggrid import AgGrid, GridOptionsBuilder



def translate_class_name(english_name):
    translations = {
        'Apple___Apple_scab': "Pommier___Tavelure du pommier",
        'Apple___Black_rot': "Pommier___Pourriture noire",
        'Apple___Cedar_apple_rust': "Pommier___Rouille du cèdre",
        'Apple___healthy': "Pommier___Sain",
        'Blueberry___healthy': "Myrtille___Sain",
        'Cherry_(including_sour)___Powdery_mildew': "Cerise (incluant griottes)___Oïdium",
        'Cherry_(including_sour)___healthy': "Cerise (incluant griottes)___Sain",
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Maïs___Taches foliaires à Cercospora",
        'Corn_(maize)___Common_rust_': "Maïs___Rouille commune",
        'Corn_(maize)___Northern_Leaf_Blight': "Maïs___Brûlure septentrionale des feuilles",
        'Corn_(maize)___healthy': "Maïs___Sain",
        'Grape___Black_rot': "Vigne___Pourriture noire",
        'Grape___Esca_(Black_Measles)': "Vigne___Esca (Maladie des taches noires)",
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Vigne___Brûlure des feuilles (Taches d'Isariopsis)",
        'Grape___healthy': "Vigne___Sain",
        'Non_leaves': "Image non reconnue.",
        'Orange___Haunglongbing_(Citrus_greening)': "Orange___Huanglongbing (Verdissement des agrumes)",
        'Peach___Bacterial_spot': "Pêche___Tache bactérienne",
        'Peach___healthy': "Pêche___Sain",
        'Pepper,_bell___Bacterial_spot': "Poivron___Tache bactérienne",
        'Pepper,_bell___healthy': "Poivron___Sain",
        'Potato___Early_blight': "Pomme de terre___Alternariose",
        'Potato___Late_blight': "Pomme de terre___Mildiou",
        'Potato___healthy': "Pomme de terre___Sain",
        'Raspberry___healthy': "Framboise___Sain",
        'Soybean___healthy': "Soja___Sain",
        'Squash___Powdery_mildew': "Courge___Oïdium",
        'Strawberry___Leaf_scorch': "Fraise___Brûlure des feuilles",
        'Strawberry___healthy': "Fraise___Sain",
        'Tomato___Bacterial_spot': "Tomate___Tache bactérienne",
        'Tomato___Early_blight': "Tomate___Alternariose",
        'Tomato___Late_blight': "Tomate___Mildiou",
        'Tomato___Leaf_Mold': "Tomate___Moisi des feuilles",
        'Tomato___Septoria_leaf_spot': "Tomate___Tache septorienne des feuilles",
        'Tomato___Spider_mites Two-spotted_spider_mite': "Tomate___Acarien à deux points",
        'Tomato___Target_Spot': "Tomate___Tache cible",
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Tomate___Virus de la feuille jaune en cuiller",
        'Tomato___Tomato_mosaic_virus': "Tomate___Virus de la mosaïque",
        'Tomato___healthy': "Tomate___Sain",
    }
    return translations.get(english_name, english_name)

plant_disease_info = {
    'Apple___Apple_scab': "La tavelure du pommier est une maladie fongique causant des lésions sombres sur les feuilles et les fruits, réduisant leur qualité.",
    'Apple___Black_rot': "La pourriture noire du pommier provoque des taches sombres sur les fruits et les branches, causée par un champignon.",
    'Apple___Cedar_apple_rust': "La rouille du cèdre-pommier est une maladie fongique nécessitant des cèdres pour compléter son cycle.",
    'Apple___healthy': "Ce pommier est en bonne santé, sans signe de maladie.",
    'Blueberry___healthy': "Cette myrtille est en bonne santé.",
    'Cherry_(including_sour)___Powdery_mildew': "L'oïdium des cerises est une maladie fongique provoquant un dépôt poudreux blanc sur les feuilles.",
    'Cherry_(including_sour)___healthy': "Ces cerises sont en bonne santé.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Les taches foliaires à Cercospora affectent les feuilles du maïs, réduisant la photosynthèse et les rendements.",
    'Corn_(maize)___Common_rust_': "La rouille commune du maïs est une maladie fongique causant des pustules rouges sur les feuilles.",
    'Corn_(maize)___Northern_Leaf_Blight': "La brûlure septentrionale des feuilles est une maladie causée par un champignon, affectant les feuilles de maïs.",
    'Corn_(maize)___healthy': "Ce maïs est en bonne santé.",
    'Grape___Black_rot': "La pourriture noire de la vigne cause des lésions brunes et des fruits pourris, réduisant la production.",
    'Grape___Esca_(Black_Measles)': "L'Esca de la vigne provoque des taches noires sur les feuilles et des pertes de production.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "La brûlure des feuilles de vigne entraîne des taches brunes et la défoliation des plants.",
    'Grape___healthy': "Cette vigne est en bonne santé.",
    'Non_leaves': "L'image que vous avez sélectionné ne correspond peut-être à une feuilles.",
    'Orange___Haunglongbing_(Citrus_greening)': "Le Huanglongbing est une maladie bactérienne grave provoquant le verdissement des agrumes et la chute des fruits.",
    'Peach___Bacterial_spot': "La tache bactérienne du pêcher cause des lésions sur les fruits et les feuilles.",
    'Peach___healthy': "Ce pêcher est en bonne santé.",
    'Pepper,_bell___Bacterial_spot': "La tache bactérienne du poivron provoque des taches brunes sur les feuilles et les fruits.",
    'Pepper,_bell___healthy': "Ce poivron est en bonne santé.",
    'Potato___Early_blight': "L'alternariose de la pomme de terre provoque des taches brunes en cibles sur les feuilles.",
    'Potato___Late_blight': "Le mildiou de la pomme de terre est une maladie dévastatrice causant des lésions aqueuses sur les feuilles et les tubercules.",
    'Potato___healthy': "Cette pomme de terre est en bonne santé.",
    'Raspberry___healthy': "Cette framboise est en bonne santé.",
    'Soybean___healthy': "Ce soja est en bonne santé.",
    'Squash___Powdery_mildew': "L'oïdium de la courge provoque un dépôt blanc poudreux sur les feuilles.",
    'Strawberry___Leaf_scorch': "La brûlure des feuilles de fraise entraîne un dessèchement et une décoloration des feuilles.",
    'Strawberry___healthy': "Cette fraise est en bonne santé.",
    'Tomato___Bacterial_spot': "La tache bactérienne de la tomate cause des lésions brunes sur les feuilles et les fruits.",
    'Tomato___Early_blight': "L'alternariose de la tomate provoque des taches circulaires brunes sur les feuilles et les fruits.",
    'Tomato___Late_blight': "Le mildiou de la tomate cause des lésions aqueuses sur les feuilles et les fruits.",
    'Tomato___Leaf_Mold': "Le moisi des feuilles de tomate provoque des taches jaunes sur le dessus des feuilles.",
    'Tomato___Septoria_leaf_spot': "La tache septorienne des feuilles de tomate cause des taches brunes avec un centre clair sur les feuilles.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Les acariens à deux points provoquent une décoloration et une chute des feuilles de tomate.",
    'Tomato___Target_Spot': "La tache cible de la tomate provoque des lésions nécrotiques sur les feuilles.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Ce virus provoque des feuilles jaunes enroulées et un ralentissement de la croissance.",
    'Tomato___Tomato_mosaic_virus': "Le virus de la mosaïque de la tomate cause des taches jaunes et une déformation des feuilles.",
    'Tomato___healthy': "Cette tomate est en bonne santé."
}

# Defining advice for each class
class_advice = {
    'Apple___Apple_scab': ": Retirez les feuilles affectées et éliminez-les correctement pour éviter leur propagation.",
    'Apple___Black_rot': ": Retirez les fruits et les feuilles infectés et traitez avec un fongicide.",
    'Apple___Cedar_apple_rust': ": Taillez les branches affectées et utilisez un fongicide.",
    'Apple___healthy': ": Votre pommier a l'air en bonne santé ! Continuez à lui prodiguer de bons soins.",
    'Blueberry___healthy': ": Votre bleuet est en bonne santé ! Assurez-lui un arrosage et un ensoleillement adéquats.",
    'Cherry_(including_sour)___Powdery_mildey': ": Traitez avec des fongicides et retirez les feuilles infectées.",
    'Cherry_(including_sour)___healthy': ": Votre cerisier est en bonne santé ! Surveillez sa croissance.",
    'Corn_(maize)___Cercospora_leaf_spot_gray_leaf_spot': ": Appliquez un fongicide et retirez les feuilles affectées feuilles.",
    'Corn_(maize)___Common_rust_': ": Utilisez un fongicide et améliorez la circulation de l'air autour de la plante.",
    'Corn_(maize)___Northern_Leaf_Blight': ": Retirez les feuilles infectées et appliquez des fongicides.",
    'Corn_(maize)___healthy': ": Votre maïs est en bonne santé ! Continuez l'irrigation et la gestion des ravageurs.",
    'Grape___Black_rot': ": Taillez les vignes infectées et appliquez des fongicides.",
    'Grape___Esca_(Black_Measles)': ": Retirez les vignes infectées et traitez avec des fongicides appropriés.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': ": Retirez les feuilles infectées et appliquez des fongicides.",
    'Grape___healthy': ": Votre vigne est en bonne santé ! Assurez un ensoleillement et un arrosage adéquats.",
    'Non_leaves': "Réessayez en sélectionnant l'image à nouveau ou une autre.",
    'Orange___Haunglongbing_(Citrus_greening)': ": Il s'agit d'une maladie grave. Retirez les arbres affectés et désinfectez les outils.",
    'Peach___Bacterial_spot': ": Taillez les branches affectées et appliquez des fongicides à base de cuivre.",
    'Peach___healthy': ": Votre pêcher est en bonne santé ! Continuez les soins réguliers.",
    'Pepper,_bell___Bacterial_spot': ": Retirez les feuilles infectées et utilisez des bactéricides.",
    'Pepper,_bell___healthy': ": Votre plant de poivron a l'air en bonne santé ! Gardez le sol bien drainé.",
    'Potato___Early_blight': ": Appliquez des fongicides et retirez les feuilles infectées.",
    'Potato___Late_blight': ": Retirez les plantes infectées et appliquez un fongicide pour empêcher toute propagation ultérieure.",
    'Potato___healthy': ": Vos plants de pommes de terre sont en bonne santé ! Assurez un bon drainage du sol.",
    'Raspberry___healthy': ": Votre plant de framboisier est en bonne santé ! Continuez les soins réguliers.",
    'Soybean___healthy': ": Votre plant de soja est en bonne santé ! Assurez un espacement approprié pour la croissance.",
    'Squash___Powdery_mildew': ": Traitez avec un fongicide et retirez les feuilles infectées.",
    'Strawberry___Leaf_scorch': ": Taillez les feuilles affectées et améliorez la gestion de l'eau.",
    'Strawberry___healthy': ": Votre plant de fraisier est en bonne santé ! Gardez le sol fertile.",
    'Tomato___Bacterial_spot': ": Taillez les zones infectées et appliquez des fongicides à base de cuivre.",
    'Tomato___Early_blight': ": Appliquez un fongicide et retirez les feuilles affectées.",
    'Tomato___Late_blight': ": Retirez les plantes infectées et appliquez des fongicides.",
    'Tomato___Leaf_Mold': ": Améliorez la ventilation autour de la plante et traitez avec des fongicides.",
    'Tomato___Septoria_leaf_spot': ": Retirez les feuilles affectées et appliquez des fongicides.",
    'Tomato___Spider_mites Two-spotted_spider_mite': ": Appliquez des acaricides et retirez les feuilles infestées.",
    'Tomato___Target_Spot': ": Appliquez des fongicides et retirez les feuilles infectées.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ": Retirez les feuilles infectées plantes et contrôlez les pucerons.",
    'Tomato___Tomato_mosaic_virus': ": Retirez les plantes infectées et désinfectez les outils.",
    'Tomato___healthy': ": Votre plante de tomates a l'air en bonne santé ! Continuez les soins réguliers."
}

# Fonction pour sauvegarder une prédiction dans un fichier CSV
def save_prediction(image_name, prediction, recommendation):
    with open('predictions_history.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), image_name, prediction, recommendation])

  
# Fonction pour afficher l'historique
def display_history():
    st.write("### Historique des prédictions")
    try:
        # Charger les données depuis le fichier CSV
        data = pd.read_csv('predictions_history.csv', header=None, names=['Date', 'Image', 'Prediction', 'Info', 'Conseils'])
        # Ajouter une option de filtrage
        search = st.text_input("Rechercher par prédiction ou recommandation", "")
        if search:
            data = data[data['Prediction'].str.contains(search, case=False) | data['Info'].str.contains(search, case=False) | data['Recommendation'].str.contains(search, case=False)]
       
        st.dataframe(data)
        
        # Bouton pour télécharger l'historique
        st.download_button("Télécharger l'historique", data.to_csv(index=False).encode('utf-8'), "historique_predictions.csv", "text/csv")
    except FileNotFoundError:
        st.write("Aucun historique disponible pour le moment.")

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras", compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Acceuil","Prédiction", "Historique", "A propos"])




#st.balloons()
#st.form( st.slider("Slider"), st.checkbox("Checkbox") )
#st.camera_input(disabled=False, label_visibility= "visible", on_change= None, label= "Photo")

# Main Page
if app_mode == "Acceuil":
    st.header("SYSTEME DE PREDICTION DES MALADIES DES PLANTES", divider="rainbow" )
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Bienvenue au système de reconnaissance des maladies des plantes !
    
    Téléversez une image d'une plante et notre système l'analysera pour détecter tout signe de maladie.
    """)

# About Project
elif app_mode == "A propos":
    st.header("A propos", divider="rainbow")
    st.markdown("""
    #### À propos des données
    Cet ensemble de données contient des images de cultures saines et malades, classées en 38 classes différentes.
    """)

# Prediction Page
elif app_mode == "Prédiction":
    st.header("Prédiction des maladies", divider='rainbow')
    
    test_image = st.file_uploader(label="Sélectionnez")
    if test_image is None:
        st.markdown(" Sélectionnez une image.")
    else:
        st.image(test_image, use_column_width=False, output_format="JPEG")
    
    if st.button("Prédire"):
        if test_image is not None:
            with st.spinner("Analyse en cours..."):
                result_index = model_prediction(test_image)
                
            # Map index to class name
            class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Non_leaves','Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']  # Full list here
            predicted_class = class_names[result_index]
            translated_class = translate_class_name(predicted_class)
            formatted_label = translated_class.replace("__", " ").replace("_", " ")
            
            
            # Récupérer les informations sur la maladie
            disease_info = plant_disease_info.get(predicted_class, "Aucune information disponible pour cette maladie.")
            disease_advice = class_advice.get(predicted_class)
                
            st.success(f"Prediction: {translated_class}")
            st.info(f"Informations sur la maladie : {disease_info}")
            st.info(f"Conseils: {disease_advice}")
            
            
            with open('predictions_history.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), test_image, translated_class, disease_info, disease_advice])
            
        else:
            st.warning("Veuillez sélectionner une image.")

elif app_mode == "Historique":
    st.header("HISTORIQUE DES PREDICTIONS", divider="rainbow")
    display_history()
    
    