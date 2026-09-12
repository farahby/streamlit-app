# Module de secours genere par la cellule de bundle (FIX E).
# Le vrai soc_chatbot.py est ecrit par la cellule CHATBOT du notebook.
import streamlit as st


def render_chatbot_tab(*args, **kwargs):
    st.warning(
        "Assistant SOC indisponible : la cellule CHATBOT du notebook n avait "
        "pas ete executee au moment de la construction du bundle. Execute-la, "
        "puis relance la cellule de push pour deployer la vraie version."
    )
    st.caption("Module de secours — aucune fonctionnalite de chat.")
