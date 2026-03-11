import streamlit as st
import main

st.title("GNN Image Graph Explorer")

st.write("Run the full AI pipeline (same as main.py)")

if st.button("Run Full Pipeline"):

    st.write("Running pipeline... please wait")

    main.main()

    st.success("Pipeline completed!")
