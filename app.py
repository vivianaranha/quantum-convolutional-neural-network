"""Interactive QCNN image-classification dashboard.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from quantum_cnn import BenchmarkConfig, run_benchmark
from quantum_cnn.data import make_single_image
from quantum_cnn.inference import predict_saved_models

st.set_page_config(page_title="Quantum Convolutional Neural Network", layout="wide")
st.title("Quantum Convolutional Neural Network")
st.caption("Created by School of AI and School of QC")
st.write(
    "Classify synthetic 8×8 vertical and horizontal line images with a shared-filter, "
    "three-stage QCNN and matched-input classical baselines."
)

with st.sidebar:
    st.header("Benchmark settings")
    samples = st.slider("Generated images", 80, 400, 120, 20)
    repeats = st.slider("Repeated splits", 1, 5, 1)
    epochs = st.slider("QCNN epochs", 5, 120, 20, 5)
    image_noise = st.slider("Image noise", 0.0, 0.4, 0.16, 0.01)
    occlusion = st.slider("Occlusion probability", 0.0, 0.8, 0.25, 0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42)
    run_button = st.button("Run benchmark", type="primary", use_container_width=True)

st.info(
    "The QCNN uses exact statevector simulation on a classical computer. The finite-shot and "
    "parameter-noise panels are sensitivity probes, not real quantum-hardware measurements."
)

if run_button:
    config = BenchmarkConfig(
        samples=samples,
        repeats=repeats,
        qcnn_epochs=epochs,
        image_noise=image_noise,
        occlusion_probability=occlusion,
        random_seed=int(seed),
    ).validate()
    status = st.empty()
    with st.spinner("Training the QCNN and classical baselines..."):
        st.session_state["qcnn_result"] = run_benchmark(config, status.write)
    status.success("Benchmark complete")

result = st.session_state.get("qcnn_result")
if result is None:
    st.subheader("Architecture")
    st.write(
        "Eight spatial patch means are angle-encoded on eight qubits. Shared quantum filters and "
        "pooling operations reduce the active topology from eight qubits to four, two, and finally "
        "one measured qubit. Every baseline receives the same eight patch features and split rows."
    )
else:
    st.subheader("Overall comparison")
    st.dataframe(
        result.summary.style.format(precision=4), hide_index=True, use_container_width=True
    )
    left, right = st.columns(2)
    with left:
        st.image(str(result.output_directory / "accuracy.png"))
        st.image(str(result.output_directory / "qcnn_training.png"))
        st.image(str(result.output_directory / "image_gallery.png"))
    with right:
        st.image(str(result.output_directory / "f1.png"))
        st.image(str(result.output_directory / "qcnn_robustness.png"))
        st.image(str(result.output_directory / "patch_features.png"))
    st.image(str(result.output_directory / "misclassifications.png"))
    with st.expander("QCNN circuit"):
        st.code(
            (result.output_directory / "qcnn_circuit.txt").read_text(encoding="utf-8"),
            language="text",
        )

    st.subheader("Try the saved first-repeat models")
    controls, preview = st.columns([1, 1])
    with controls:
        orientation = st.selectbox("Orientation", ["vertical", "horizontal"])
        position = st.slider("Line position", 0, 7, 3)
        thickness = st.select_slider("Line thickness", [1, 2], value=1)
        sample_noise = st.slider("Sample noise", 0.0, 0.5, 0.12, 0.01)
        sample_seed = st.number_input("Sample seed", min_value=0, value=7)
        sample_occluded = st.checkbox("Add occlusion")
    image = make_single_image(
        orientation,
        position,
        thickness,
        sample_noise,
        int(sample_seed),
        sample_occluded,
    )
    with preview:
        figure, axis = plt.subplots(figsize=(3, 3))
        axis.imshow(image, cmap="magma", vmin=0, vmax=1)
        axis.set_xticks([])
        axis.set_yticks([])
        st.pyplot(figure, clear_figure=True)
    if st.button("Classify image"):
        st.dataframe(
            predict_saved_models(result.output_directory, image),
            hide_index=True,
            use_container_width=True,
        )

st.divider()
st.caption("Created by School of AI and School of QC")
