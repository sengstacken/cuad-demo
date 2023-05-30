FROM python:3.7

ARG DASHBOARD_SAGEMAKER_MODEL
ENV DASHBOARD_SAGEMAKER_MODEL=$DASHBOARD_SAGEMAKER_MODEL

EXPOSE 80
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

CMD streamlit run streamlit_app.py \
    --server.headless true \
    --browser.serverAddress="0.0.0.0" \
    --server.enableXsrfProtection false \
    --server.enableCORS false \
    --server.port 80 \
    --browser.gatherUsageStats false


