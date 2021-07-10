
# CRB-CRS Source Code

We have two main directories at the root level. The first named ‘src’ contains the complete Python source code for CRB-CRS. The second directory, named ‘data’ contains the data for both movies recommendation and dialogs.

## Installation

**In order to reproduce the results of our paper, the Python scripts under the directory ‘src’ should be executed in the given order.**


1. The file named ‘PrepareTrainingData.py’ is used to parse ReDial training data into a text file based on each dialog between the seeker and human recommender.

2. The file named ‘PreprocessSentences.py’ is used to preprocess the data parsed in the previous step. This piece of script outputs a text file containing training data after prepossessing, as mentioned in the paper.

3. The file named ‘RetrieveCandidatesByHistory.py’ is used to train the TF-IDF model and retrieve candidate responses. The approaches for outliers pruning and ranking of the retrieved set of candidate responses are also implemented in the same file.

4. The file named ‘Cacululate_MLE_Probs.py’ is used by the script in the file ‘RetrieveCandidatesByHistory.py’ to calculate the bigram probabilities in our ranking technique.

5. The file named ‘IntegrateRecommendations.py’ is used for the purpose of the integration of both movies recommendation and metadata (genre, actor, etc.) information. 

6. The remaining three .py files starting from **‘Recommender’** are recommender approaches used to compute and integrate recommendations in the retrieved responses. Note that these files are used by the script in the file 'IntegrateRecommendations.py'.

7. Before running the scripts, please run ***‘Requirementes.txt’*** which includes all the Python libraries to install in order to reproduce the results.

**The description of the content in the 'data' directory is as follows.**

There are three subdirectories under this directory named ‘DialogData’ and ‘RecommendersItemData’ and ‘redial_dataset’.

  1. 	Under the ‘DialogData’ directory, we have all the dialogs’ data files used or exported using python scripts, as previously mentioned. This data is used to train the model and retrieve candidate responses.

  2. 	Under the ‘RecommendersItemData’ directory, we have the MovieLens rating dataset, which is used to compute recommendations and integrate metadata information (e.g., genres, actors, etc.) in the retrieved responses. For the purpose of the model’s offline initialization, we converted the original MovieLens data into further files.

  3. The directory named redial_dataset.zip contains the original ***ReDial*** dataset split in both training and test sets.

The file named ‘KBRD & KGSF.pdf’ contains the details on how to access the source code of both the ***KBRD*** and ***KGSF*** systems.


```bash
Note: The root folder named ‘RB-CRS Code’ can be imported as a standalone Python project.
```
