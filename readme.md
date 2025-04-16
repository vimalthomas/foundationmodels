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
└── (llm_project.ipynb)              #Jupyter notebooks for  orchestration and hyperparameter tuning

