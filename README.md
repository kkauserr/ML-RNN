**Bitcoin Price Prediction with RNN (PyTorch)
**
This repository contains my solution for Lab 4 from my upper-division machine learning course. The assignment focuses on building and training a Recurrent Neural Network (RNN / LSTM / GRU) in PyTorch to perform a regression task: predicting the daily closing price of Bitcoin based on historical market data.

The lab emphasizes understanding PyTorch datasets and dataloaders, working with sequential data, and tuning model architectures and hyperparameters to achieve strong predictive performance — specifically an R² score ≥ 0.80.

**📊 Task Overview
**
Input data: coin_Bitcoin.csv

**Features used:
**
- High
- Low
- Open

**Target variable:
**
- Close

Goal: Train an RNN-based model to predict the closing price

**Evaluation metric:
**
- R² score (must be stored in a variable named r2score)

This is strictly a regression problem — not classification.

**🧠 Model
**
You may use:

✔ Standard RNN
✔ LSTM
✔ GRU

I implemented my model in PyTorch 1.13.0, keeping the architecture intentionally compact so it runs efficiently on CPU-only grading hardware.

**Key learning objectives included:
**
- Building custom datasets & dataloaders
- Handling sequential financial data
- Avoiding over-engineering
- Tuning hyperparameters for performance
- Achieving reproducible training

The final model is under ~120 lines of core logic.

**📁 File Structure
**.
├── rnn.py              # main implementation file
├── coin_Bitcoin.csv    # dataset (provided in Gradescope environment)
└── README.md           # project documentation


**🛠️ Tech & Dependencies
**
Allowed packages:

- torch==1.13.0
- torchvision==0.14.0
- numpy==1.23.2
- pandas==1.4.4
- sklearn==1.2.0

**📈 Performance Requirement
**
Your final model must achieve:

**R² ≥ 0.80
**
and store the value in:

**r2score = ...**

**⚠️ Academic Policy**
- Code may resemble PyTorch documentation examples
- Any non-official tutorials MUST be cited in comments
- Overly-large models (e.g., 1024-unit layers) are discouraged due to CPU limits
- Runtime must complete in under 10 minutes
- The lab is designed to be simple & readable — not over-engineered


**🚀 Running the Code
**python rnn.py


**The script will:
**
- Load and preprocess the dataset
- Create PyTorch dataset/dataloader
- Train the RNN model
- Evaluate on test data
- Print and store the R² score

**🙏 Acknowledgements
**
This project was completed as part of a 4xx-level machine learning course.
PyTorch tutorials and documentation were used as reference material.
