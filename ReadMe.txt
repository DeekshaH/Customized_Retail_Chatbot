To execute Customised Chatbot which answers only retail store question answers with mistral 7b model code follow

this simple Steps:

create new Conda Environment from command prompt, make sure you have Anaconda installed

commands to create new environment:

1.mkdir <dirname>

2.cd <dirname>

3.conda create-name <envname> python

4.conda activate <envname>

Now you are good with environment setting, you can install all the necessary packages.

1.pip install streamlit

2.pip install langchain_community

3.pip install huggingface_hub

Make Sure you have downloaded Mistral model locally and kept in the same folder. downloaded mistral model is in my chatbot repository do check it out for downloading.

run python file

streamlit run main.py