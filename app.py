import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import os

# Ensure Matplotlib doesn't use a GUI backend to prevent Streamlit threading issues
import matplotlib
matplotlib.use("Agg")

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pharyngitis Detection AI", layout="wide")

IMG_SIZE = (224, 224)
CLASSES = ['Normal', 'Pharyngitis']

# Update these URLs with the raw download links to your hosted .h5 files
# (e.g., from GitHub Releases or Hugging Face Hub)
MODEL_URLS = {
    'mobilenetv2_final.h5': 'sha256:8b5e18f4f182bab66b96120f4cf1a8143d878c96a0ff57db18b59ec32ddf2514',
    'efficientnetb3_final.h5': 'sha256:216aea6de31bb59e5d2637eb3a408470503a4cbb2f136263f5f7bfa47234c675',
    'resnet50v2_final.h5': 'sha256:dc12bf1c0c42ef8ddaf4f8be7a4ad3984deedcc972c3363ea463be010717eecd',
    'densenet121_final.h5': 'sha256:e897a807c78742999479178c133366338338d71d30db5aa891be36ee521c9ef3'
}

# Preprocessing map based on Keras applications
PREPROCESS_MAP = {
    'mobilenetv2': keras.applications.mobilenet_v2.preprocess_input,
    'efficientnetb3': keras.applications.efficientnet.preprocess_input,
    'resnet50v2': keras.applications.resnet_v2.preprocess_input,
    'densenet121': keras.applications.densenet.preprocess_input
}

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

def focal_loss(gamma=2.0, alpha=0.25):
    """Custom focal loss closure exactly as defined in the project."""
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        ce = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        return tf.reduce_mean(
            tf.reduce_sum(alpha * tf.pow(1.0 - p_t, gamma) * ce, axis=-1)
        )
    return loss_fn

@st.cache_resource(show_spinner="Downloading and loading models (this may take a minute on first run)...")
def load_ensemble_models():
    """Downloads models from URLs if not cached locally, then loads them safely."""
    models = {}
    for name, url in MODEL_URLS.items():
        try:
            # keras.utils.get_file will download and cache the file in ~/.keras/models/
            file_path = keras.utils.get_file(
                fname=name,
                origin=url,
                cache_subdir='models'
            )
            # Custom object scope for the specific focal_loss implementation
            model = keras.models.load_model(
                file_path, custom_objects={'loss_fn': focal_loss()}
            )
            models[name] = model
        except Exception as e:
            st.error(f"Failed to load {name}. Please check the URL. Error: {e}")
    return models

def get_preprocessing_function(model_name):
    """Resolves the preprocessing function using the filename substring."""
    name_lower = model_name.lower()
    for key, func in PREPROCESS_MAP.items():
        if key in name_lower:
            return func
    # Fallback default if name doesn't explicitly match
    return keras.applications.imagenet_utils.preprocess_input

def generate_grad_cam(img_array, model, class_index, layer_name=None):
    """Generates Grad-CAM heatmap for a given image and model."""
    # Find the last conv layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                layer_name = layer.name
                break
                
    if layer_name is None:
        return None # Could not find a conv layer

    grad_model = keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.title("🩺 Automated Pharyngitis Detection")
st.markdown("""
This system uses a soft-voting ensemble of four deep learning backbones (MobileNetV2, EfficientNetB3, 
ResNet50V2, and DenseNet121) to detect pharyngitis from throat photographs.
""")

# Automatically load models on startup
models = load_ensemble_models()

if len(models) == 4:
    st.sidebar.success("✅ All 4 ensemble models loaded successfully from the cloud.")
else:
    st.sidebar.warning(f"⚠️ Only {len(models)}/4 models loaded. Check the URLs in the code.")

st.header("Patient Evaluation")
uploaded_img = st.file_uploader("Upload Throat Photograph (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_img is not None and len(models) == 4:
    # Read and resize image
    image = Image.open(uploaded_img).convert('RGB')
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, IMG_SIZE)
    
    col1, col2 = st.columns()
    
    with col1:
        st.image(image_resized, caption="Uploaded Image", use_container_width=True)
        analyze_btn = st.button("Analyze Image", type="primary", use_container_width=True)
        
    if analyze_btn:
        with st.spinner("Processing through 4-model ensemble..."):
            all_probs = []
            model_names = []
            efficientnet_model = None
            efficientnet_img_batch = None
            
            # 1. Run Predictions
            for fname, model in models.items():
                preprocess_fn = get_preprocessing_function(fname)
                img_batch = preprocess_fn(np.expand_dims(image_resized.astype(np.float32), axis=0))
                
                probs = model.predict(img_batch, verbose=0)
                all_probs.append(probs)
                model_names.append(fname.split('_')) # simplify name
                
                # Save efficientnet input for Grad-CAM
                if 'efficientnet' in fname.lower():
                    efficientnet_model = model
                    efficientnet_img_batch = img_batch
            
            # 2. Ensemble Aggregation (Soft-Voting)
            ensemble_probs = np.mean(all_probs, axis=0)
            pred_idx = np.argmax(ensemble_probs)
            confidence = ensemble_probs[pred_idx]
            severity = ensemble_probs * 100  # Pharyngitis probability * 100
            pred_class = CLASSES[pred_idx]
            
            # 3. Clinical Recommendation Logic
            if pred_class == 'Normal' and severity < 30:
                rec = "Maintain standard throat hygiene. Warm saline gargles are advisable. Monitor symptoms if mild discomfort persists."
                tier_color = "green"
            elif 30 <= severity < 70:
                rec = "Rest and maintain adequate fluid intake. Non-prescription analgesics may relieve discomfort. Re-evaluate in 48-72 hours if symptoms worsen."
                tier_color = "orange"
            else:
                rec = "Seek prompt in-person medical consultation. Rapid antigen testing is recommended. Antibiotic therapy only if bacterial aetiology is confirmed."
                tier_color = "red"
                
            # 4. Display Results
            st.divider()
            st.subheader("Diagnostic Results")
            
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Prediction", pred_class)
            r_col2.metric("Ensemble Confidence", f"{confidence:.1%}")
            r_col3.metric("Severity Score", f"{severity:.1f}/100")
            
            if confidence < 0.70:
                st.warning("⚠️ Uncertainty Warning: Ensemble confidence is below 70%. Physical examination is strongly recommended.")
                
            st.markdown(f"**Clinical Recommendation:** :{tier_color}[{rec}]")
            
            # 5. Visualizations
            st.subheader("Ensemble Breakdown & Interpretability")
            v_col1, v_col2 = st.columns(2)
            
            # Panel 2: Per-Backbone Confidence Bar Chart
            with v_col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                y_pos = np.arange(len(model_names))
                pharyngitis_probs = [p * 100 for p in all_probs]
                
                bars = ax.barh(y_pos, pharyngitis_probs, align='center', color='skyblue')
                ax.set_yticks(y_pos, labels=model_names)
                ax.invert_yaxis()  # labels read top-to-bottom
                ax.set_xlabel('Pharyngitis Probability (%)')
                ax.set_title('Individual Backbone Predictions')
                ax.set_xlim(0, 100)
                
                # Add values to bars
                for bar in bars:
                    width = bar.get_width()
                    ax.annotate(f'{width:.1f}%',
                                xy=(width, bar.get_y() + bar.get_height() / 2),
                                xytext=(3, 0),  # 3 points horizontal offset
                                textcoords="offset points",
                                ha='left', va='center')
                st.pyplot(fig)
            
            # Panel 3: Grad-CAM (using EfficientNetB3 as per document)
            with v_col2:
                if efficientnet_model is not None:
                    heatmap = generate_grad_cam(efficientnet_img_batch, efficientnet_model, pred_idx)
                    if heatmap is not None:
                        # Resize heatmap to match image width/height
                        heatmap_resized = cv2.resize(heatmap, (image_resized.shape, image_resized.shape))
                        heatmap_resized = np.uint8(255 * heatmap_resized)
                        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                        
                        # Superimpose
                        superimposed_img = cv2.addWeighted(image_resized, 0.6, heatmap_colored, 0.4, 0)
                        
                        st.image(superimposed_img, caption="Grad-CAM Visualization (EfficientNetB3)", use_container_width=True)
                else:
                    st.info("EfficientNetB3 model not found in the ensemble; skipping Grad-CAM.")

            st.caption("🛑 **DISCLAIMER:** This tool provides automated decision support only and is not a substitute for professional clinical evaluation.")

elif uploaded_img is not None:
    st.warning("Models are currently downloading or failed to load. Please wait or check model URLs.")
