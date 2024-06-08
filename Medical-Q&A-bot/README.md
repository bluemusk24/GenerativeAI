Steps to run the project

* create a virtual environment medic-env
```bash
conda create -n medic-env python=3.10
```

* Activate a virtual environment
```bash
conda activate medic-env 
```

* create requirements.txt file
```bash
touch requirements.txt
```

* paste and view the necessary libraries for this project in requirements text file
```bash
nano requirements.txt
cat requirements.txt                        
```

* install requirements file
```bash
pip install -r requirements.txt
```

* download the llama-2-7b-chat model 'llama-2-7b-chat.Q4_K_M.gguf' version below:

```ini
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
```

* jupyter notebook file
```bash
medic-chatbot.ipynb
```

### Create a '.env' file in the root directory and add your pinecone credentials as follows:

```ini
os.environ['PINECONE_API_KEY'] = "*********************"
```

```bash
python store_index.py
```

```bash
python app.py
```

Now,
```bash
open up locahost:
```

### Techstack used

- Python
- Langchain
- Flask
- Meta llama
- Pinecone
