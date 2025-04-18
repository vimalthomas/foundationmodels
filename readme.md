project strcuture:

llm_project/

├── dataset_loader.py         # Dataset class and collate function collection

├── gru_model.py              # GRU language model script

├── lstm_model.py             # LSTM language model script

├── rnn_model.py              # Vanilla RNN language model script

├── transformer_model.py      # Transformer language model script

├── tokenizer.py              # Tokenizer wrapper which includes data downloader & trainer

├── train_utils.py            # Training, evaluation, and plotting utils

├── README.md                 # This file

└── llm_project.py           #Jupyter notebooks convereted py file. 

└── llm_experiment.py           #Jupyter notebooks convereted py file used for runnning model experiments.(make sure the corpus.txt, train.jsonl and test.jsonl are already prpesent to run this py file).


How to run the project?

Place all the files in a source directory and run the llm_project.py file. This will run and generate the models. Again, if we need to run the experiments, run the llm_experiment.py file.



