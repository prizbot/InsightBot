# InsightBot

## Overview
InsightBot is a comprehensive company insights chatbot that provides detailed information about companies, including ratings, reviews, locations, and employee counts. The application can be run locally using Streamlit or as a server-client architecture for high-performance requirements.

## Directory Structure
- **data/**: Contains the dataset file.
  - `companyreview_dataset.csv`
- **models/**: Contains model-related files.
  - `instruction.txt`
- **vectorstore/**: Contains the FAISS index files.
  - `db_faiss/`
    - `index.faiss`
    - `index.pkl`
- **server_client/**: Contains server and client scripts for running the application in a server-client setup.
  - `server.py`
  - `client.py`
- **script.py**: Runs the application locally using Streamlit.
- **requirements.txt**: Lists all the dependencies required for the project.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/InsightBot.git
   cd InsightBot
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt

## Usage
### Running Locally with Streamlit:
1. Execute the script.py:
   ```sh
   streamlit run script.py
### Running with Server-Client Setup:
1. Start the server:
   ```sh
   uvicorn server_client.server:app --host 0.0.0.0 --port 8000
2. Run the client on the desired system:
   ```sh
   python server_client.client.py

## Notes
1. Ensure the server system has a high GPU for optimal performance.
2. Modify the server IP and port in client.py as needed.

## Conclusion

This repository contains code for InsightBot, a comprehensive company insights chatbot powered by LangChain, Streamlit, and FastAPI. It allows users to query information about companies from a dataset using conversational retrieval techniques.

### For more details:

Visit the [project repository](https://github.com/prizbot/InsightBot).

### Contact Information:

For any inquiries or collaborations, feel free to reach out:
- LinkedIn: [Priyadharshini NRS](https://www.linkedin.com/in/priyadharshininrs)
- Email: [priyadharshininrs@gmail.com](mailto:priyadharshininrs@gmail.com)

