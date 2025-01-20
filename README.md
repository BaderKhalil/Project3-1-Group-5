# Mining Misconceptions in Mathematics
Project 3-1 Group 5

This project provides classification models that predicts misconceptions from multiple choice questions' wrong answers.

Our final model is located in the `models\RF-Complete.ipynb` notebook. The other notebooks were used to perform experiments for our report.
To run the file, the following libraries are needed:
```
pip install numpy
pip install pandas
pip install sentence-transformers
pip install scikit-learn
pip install torch torchvision torchaudio
pip install imbalanced-learn
```

The *.csv files in `models\datasets\eedi-mining-misconceptions-in-mathematics` were obtained by joining a [Kaggle competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data) and adhere to the linked [license](https://creativecommons.org/licenses/by-nc/4.0/).  
The *.csv files in `models\datasets\eedi-external-dataset` were obtained via a participant of previously mentioned Kaggle competition [here](https://www.kaggle.com/datasets/alejopaullier/eedi-external-dataset).

Note: if you are seeing this project on GitHub, the *.safetensors files for the fine tuning could not be uploaded due to file size reasons, therefore making the fine-tuned models in the code to be unable to run. If you want access to the *.safetensors files, do contact one of the following emails:
- e.saussoy@student.maastrichtuniversity.nl
- m.tugay@student.maastrichtuniversity.nl
- n.gurmuzachi@student.maastrichtuniversity.nl
