from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import streamlit as st
from transformers.pipelines import pipeline
import json
import sys
#from predict import run_prediction
import random
from io import StringIO
import requests
import boto3 

st.set_page_config(layout="wide")

# grab environment variables
ENDPOINT_NAME = 'TestCUADEndPoint'

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
    
    # load questions
    with open('./data/questions.txt','r') as txt:
        data = txt.readlines()
    questions = [s.rstrip() for s in data]

    # load showquestions
    with open('./data/showquestions.txt','r') as txt:
        data = txt.readlines()
    showquestions = [s.rstrip() for s in data]
    
    return questions, showquestions

def load_contracts():
    with open('./data/test.json') as json_file:
        data = json.load(json_file)

    contracts = []
    for i, q in enumerate(data['data']):
        contract = ' '.join(data['data'][i]['paragraphs'][0]['context'].split())
        contracts.append(contract)
    return contracts

def show_contract():
    # read contract data
    if st.session_state.contract != 'Surprise Me!':
        # read in file
        # show contract
        idx = contracts_drop.index(st.session_state.contract)
        with open('./data/'+contracts_files[idx-1]) as f:
            st.session_state.contract_data = f.read()
    else:
        st.session_state.contract_data = contracts[random.randint(0,len(contracts)-1)]

    return

questions, showquestions = load_questions()
contracts = load_contracts()

# Initialization
if 'contract_data' not in st.session_state:
    st.session_state['contract_data'] = contracts[random.randint(0,len(contracts)-1)]

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
contract = st.sidebar.selectbox('Please Select a Contract',contracts_drop,key='contract',on_change=show_contract)

# upload contract
user_upload = st.sidebar.file_uploader('Please upload your own',type=['docx','pdf','txt'],accept_multiple_files=False)

# options
st.sidebar.header("Options")
# number of results
numresults = st.sidebar.selectbox('Number of Results to Show',[1,2,3,4,5],index=2)
# probabilities
show_probs = st.sidebar.checkbox('Show Probabilities', value=False)

# highlight text
#highlight = st.sidebar.checkbox('Highlight Results', value=False)

# process upload
if user_upload is not None:
    print(user_upload.name,user_upload.type)
    extension = user_upload.name.split('.')[-1].lower()
    if extension == 'txt':
        print('text file uploaded')
         # To convert to a string based IO:
        stringio = StringIO(user_upload.getvalue().decode("utf-8"))
        
        # To read file as string:
        st.session_state.contract_data = stringio.read()
    
    elif extension == 'pdf':
        import PyPDF4
        try:
            # Extracting Text from PDFs
            pdfReader = PyPDF4.PdfFileReader(user_upload)
            print(pdfReader.numPages)
            contract_temp = ''
            for i in range(0,pdfReader.numPages):
                
                print(i)
                pageobj = pdfReader.getPage(i)
                st.session_state.contract_data = contract_temp + pageobj.extractText()
        except:
            st.warning('Unable to read PDF, please try another file')

    elif extension == 'docx':
        import docx2txt 

        st.session_state.contract_data = docx2txt.process(user_upload)

    else:
        st.warning('Unknown uploaded file type, please try again')
    
### DEFINE MAIN PAGE
st.header("Legal Contract Review Demo")
st.write("This demo uses the CUAD dataset for Contract Understanding.")

paragraph = st.text_area(label="Contract",value=st.session_state.contract_data.encode('utf-8').decode('unicode-escape'),height=350)
question = st.selectbox('Choose one of the 41 queries from the CUAD dataset:', showquestions)

if st.button('Analyze'):
    if (not len(paragraph)==0) and not (len(question)==0):
        print('getting predictions')
        with st.spinner(text='Analysis in progress...'):
            #predictions = run_prediction([question], paragraph, '../models/roberta-base/')
            data = {}
            data['question'] = [questions[int(question.split(':')[0].split('Q')[-1])-1]]
            data['context'] = paragraph
            data['nbest'] = numresults
            predictions=predict(data)
            print(numresults)
            print(predictions['0'])
        if numresults==1:
            if predictions['0'][0]['text'] == "":
                answer = 'No answer found in document'
            else:
                answer = predictions['0'][0]['text']
            if show_probs:
                answer += f"  {predictions['0'][0]['probability']:.2%}"
        else:
            answer = ''
            for i in range(numresults):
                if show_probs:
                    answer += f"------ RESULT #{i+1}:   {predictions['0'][i]['probability']:.2%} ------\n"
                else:
                    answer += f'------ RESULT #{i+1} ------\n'
                if predictions['0'][i]['text'] == "":
                    answer += 'No answer found'
                else:
                    answer += predictions['0'][i]['text']
                answer +='\n\n'

        st.text_area(label="Answer",value=f"{answer}")
        st.success("Successfully processed contract!")
    else:
        st.write("Unable to call model, please select question and contract")
    