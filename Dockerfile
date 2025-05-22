#use official python image
FROM Python:3.12.4-slim

#set working directory
WORKDIR/chatbot

#Install system Dependencies
RUN apt-get update
