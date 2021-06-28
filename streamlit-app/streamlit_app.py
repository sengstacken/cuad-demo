from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import streamlit as st
from transformers.pipelines import pipeline
import json
import sys
from predict import run_prediction
import random
from io import StringIO
import requests
import boto3 

st.set_page_config(layout="wide")
st.cache(show_spinner=False, persist=True)

# grab environment variables
ENDPOINT_NAME = 'pytorch-inference-2021-06-08-19-04-51-110'

def predict(data):

    sagemaker_client = boto3.client('sagemaker-runtime', region_name='us-east-1')
    try:
        response = sagemaker_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME, 
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(data,ensure_ascii=False)
        )
    except sagemaker_client.exceptions.ClientError as e:
        if "ExpiredTokenException" in str(e):
            raise Exception("""
                ExpiredTokenException.
                You can refresh credentials by restarting the Docker container.
                Only occurs during development (due to passing of temporary credentials).
            """)
        else:
            raise e
    body_str = response['Body'].read().decode("utf-8")
    body = json.loads(body_str)
    return body



def load_questions():
	with open('../data/test.json') as json_file:
		data = json.load(json_file)

	questions = []
	for i, q in enumerate(data['data'][0]['paragraphs'][0]['qas']):
		question = data['data'][0]['paragraphs'][0]['qas'][i]['question']
		questions.append(question)
	return questions


st.cache(show_spinner=False, persist=True)
def load_contracts():
	with open('../data/test.json') as json_file:
		data = json.load(json_file)

	contracts = []
	for i, q in enumerate(data['data']):
		contract = ' '.join(data['data'][i]['paragraphs'][0]['context'].split())
		contracts.append(contract)
	return contracts


questions = load_questions()
contracts = load_contracts()

### DEFINE SIDEBAR
st.sidebar.title("Interactive Contract Analysis")
st.sidebar.markdown(
"""
Process text with [Huggingface](https://huggingface.co) models and visualize the results.  

This model uses a pretrained snapshot trained on the [Atticus](https://www.atticusprojectai.org/) Dataset - CUAD


"""
)

st.sidebar.header("Contract Selection")

# select contract
contracts_drop = ['Surprise Me!','contract 1', 'contract 2', 'contract 3']
contracts_files = ['contract-1.txt','contract-2.txt','contract-3.txt']
contract = st.sidebar.selectbox('Please Select a Contract',contracts_drop)

# read contract data
if contract != 'Surprise Me!':
	# read in file
	# show contract
	idx = contracts_drop.index(contract)
	with open('../data/'+contracts_files[idx-1]) as f:
		contract_data = f.read()
else:
	contract_data = contracts[random.randint(0,len(contracts)-1)]

# upload contract
user_upload = st.sidebar.file_uploader('Please upload your own',type=['docx','pdf','txt'],accept_multiple_files=False)
print(user_upload)

# process upload
if user_upload is not None:
	print(user_upload.name,user_upload.type)
	extension = user_upload.name.split('.')[-1].lower()
	if extension == 'txt':
		print('text file uploaded')
		 # To convert to a string based IO:
		stringio = StringIO(user_upload.getvalue().decode("utf-8"))
		
		# To read file as string:
		contract_data = stringio.read()
	
	elif extension == 'pdf':
		import PyPDF4
		try:
			# Extracting Text from PDFs
			pdfReader = PyPDF4.PdfFileReader(user_upload)
			print(pdfReader.numPages)
			contract_data = ''
			for i in range(0,pdfReader.numPages):
				
				print(i)
				pageobj = pdfReader.getPage(i)
				contract_data = contract_data + pageobj.extractText()
		except:
			st.warning('Unable to read PDF, please try another file')

	elif extension == 'docx':
		import docx2txt 

		contract_data = docx2txt.process(user_upload)

	else:
		st.warning('Unknown uploaded file type, please try again')
	


### DEFINE MAIN PAGE
st.header("Legal Contract Review Demo")
st.write("This demo uses the CUAD dataset for Contract Understanding.")

question = st.selectbox('Choose one of the 41 queries from the CUAD dataset:', questions)
paragraph = st.text_area(label="Contract",value=contract_data,height=400)
if st.button('Analyze'):
	if (not len(paragraph)==0) and not (len(question)==0):
		print('getting predictions')
		with st.spinner(text='Analysis in progress...'):
			#predictions = run_prediction([question], paragraph, '../models/roberta-base/')
			data = {}
			data['question']=[question]
			data['context']=paragraph
			print(data)
			#print(json.dumps(data))
			#resp = requests.post('https://v9sobmk44a.execute-api.us-east-1.amazonaws.com/test',json=data)
			predictions=predict(data)
			# print(resp)
			# predictions=resp.json()
			print(predictions)
		if predictions['0'] == "":
			answer = 'No answer found in document'
		else:
			answer = predictions['0']
		st.text_area(label="Answer",value=f"{answer}")
		st.success("Successfully processed contract!")
	else:
		st.write("Unable to call model, please select question and contract")
	


	