import streamlit as st


def main():
    st.set_page_config(page_title="Ask My Docs", page_icon=":books:")

    st.header("Ask My Docs")
    st.text_input("Ask a question about your documents")

    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Upload your pdfs here and click on Process")
        st.button("Process")


if __name__ == '__main__':
    main()