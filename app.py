import os

import streamlit as st

import main


st.set_page_config(page_title="Few-Shot GNN Image Graph Explorer", layout="wide")

st.title("Few-Shot GNN Image Graph Explorer")
st.write("Run the few-shot image graph pipeline for the mini VAIPE dataset.")

if st.button("Run Full Pipeline"):
    with st.spinner("Running pipeline..."):
        main.main()
    st.success("Pipeline completed!")

    if os.path.exists("outputs"):
        st.subheader("Generated outputs")
        for file_name in sorted(os.listdir("outputs")):
            st.write(f"- outputs/{file_name}")
