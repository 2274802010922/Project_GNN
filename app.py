import os

import streamlit as st

import main


st.set_page_config(page_title="GNN Image Graph Explorer", layout="wide")

st.title("GNN Image Graph Explorer")
st.write("Run the full image-graph pipeline for the mini VAIPE dataset.")

if st.button("Run Full Pipeline"):
    with st.spinner("Running pipeline..."):
        main.main()
    st.success("Pipeline completed!")

    if os.path.exists("outputs"):
        st.subheader("Generated outputs")
        output_files = sorted(os.listdir("outputs"))
        for file_name in output_files:
            st.write(f"- outputs/{file_name}")
