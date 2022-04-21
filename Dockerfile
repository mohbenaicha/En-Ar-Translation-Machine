FROM python:3.9.5

RUN adduser --disabled-password --gecos '' api-user

WORKDIR /translator-api


ADD ./dockerized-api /translator-api/
RUN pip install --upgrade pip
RUN pip install -r /translator-api/requirements.txt

# execute and ownership permssions
RUN chmod +x /translator-api/run_app.sh
RUN chown -R api-user:api-user ./

USER api-user

EXPOSE 8001

CMD ["bash", "./run_app.sh"]
