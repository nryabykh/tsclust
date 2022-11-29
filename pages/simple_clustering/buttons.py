import streamlit as st


def set_btn_clicked(btn_name: str):
    st.session_state[btn_name] = True
