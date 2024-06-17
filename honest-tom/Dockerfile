FROM python:3.11-slim
WORKDIR /app
ENV FLASK_APP=cars.py

# Copy and install dependencies
COPY dependencies.txt .
RUN pip install --no-cache-dir -r dependencies.txt
COPY cars.py .
COPY data.csv .
COPY CreditScoring.csv .

EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]