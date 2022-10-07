import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import requests
from lime import lime_tabular
from matplotlib import pyplot as plt
#from sklearn.neighbors import KneighborsClassifier
import plotly.graph_objects as go
import plotly.figure_factory as ff
pickled_model=joblib.load('model.sav')

data=pd.read_json(r'data.json')
predict_data=pd.read_json(r'predict_df.json')
predict_data=predict_data.set_index('index')

explainer = lime_tabular.LimeTabularExplainer(data.drop('TARGET',axis=1).values, mode="classification",
                                              class_names=[0,1],
                                              feature_names=data.columns)
def lime(client_id):
	
    explanation = explainer.explain_instance(predict_data.loc[client_id,:].values, pickled_model.predict_proba,
    	                                      num_features=len(predict_data.columns))

    return(explanation)





header=st.container()
client_score=st.container()
feature_importance_local=st.container()
feature_graph=st.container()
featur_importance_global=st.container()
client_info=st.container()
modeltraining=st.container()
neighbors=st.container()
def request(client_number):
    url = 'https://loan-default-api-1.herokuapp.com/process_json'
    r = requests.get(url,json={'client_num':client_number})
    return r.json()


with header:
	st.title('Prêt à dépenser')
	client_num=st.selectbox('Select client number',predict_data.index)


with feature_importance_local:
     st.header('Local Explanation')	
     if st.button("Explain Results"):

        with st.spinner('Calculating...'):

	        components.html(lime(client_num).as_html(), height=800)
	#st.pyplot(lime(client_num).as_pyplot_figure())
	#plt.clf()
    #st.markdown(lime(client_num).as_html(), unsafe_allow_html=True)
with client_score:
	  fig=go.Figure(go.Indicator(mode="gauge+number",value=request(client_num),title={'text':'Score'}))
	  st.header('Prediction')
	  st.plotly_chart(fig,use_container_width=True)
	#st.write(request(client_num))
	  


with feature_graph:
	 st.header('Feature vs Target bar chart')
	 inp_col,out_col=st.columns(2)
with inp_col:
     st.header('Select Feature') 	 	
     inp1=st.selectbox('Select Feature 1',data.columns)
     inp2=st.selectbox('Select Feature 2',data.columns)
with out_col:
     #hist_data=[list(data[data['TARGET']==0].head(100)[inp].values)]
     #fig=ff.create_distplot(hist_data,group_labels='0')
     st.header('Feature Graphs')
     fig, axs = plt.subplots(2,2,figsize=(20,20))
     axs[0,0].set_title('Distribution de'+inp1+'pour les crédits acceptés')
     axs[1,0].set_title('Distribution de'+inp1+'pour les crédits refusés')
     axs[0,1].set_title('Distribution de'+inp2+'pour les crédits acceptés')
     axs[1,1].set_title('Distribution de'+inp2+'pour les crédits refusés')
     axs[0,0].hist(x=inp1, density=False, color='turquoise', ec='blue', data=data[data['TARGET']==0])
     axs[1,0].hist(x=inp1, density=False, color='turquoise', ec='blue', data=data[data['TARGET']==1])
     axs[0,0].axvline(x=predict_data.loc[client_num,inp1], color='red')
     axs[1,0].axvline(x=predict_data.loc[client_num,inp1], color='red')
     axs[0,1].hist(x=inp2, density=False, color='turquoise', ec='blue', data=data[data['TARGET']==0])
     axs[1,1].hist(x=inp2, density=False, color='turquoise', ec='blue', data=data[data['TARGET']==1])
     axs[0,1].axvline(x=predict_data.loc[client_num,inp2], color='red')
     axs[1,1].axvline(x=predict_data.loc[client_num,inp2], color='red')
     #st.plotly_chart(fig,use_container_width=True)

     st.pyplot(fig) 
with neighbors:
     st.header('10 Similar clients')









