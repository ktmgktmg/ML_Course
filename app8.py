# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:17:22 2024

@author: thein
"""


import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:     show_explore_page()
