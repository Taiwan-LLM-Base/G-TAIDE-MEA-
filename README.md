# G-TAIDE (MEA)

FineTune mea data on llama3

- preprocess_data.py: transform the data to llama3 format text for huggingface trainer.
- train_mea.py: train the model use V100\*8. If H100 is availabe, use BF16 is better.
