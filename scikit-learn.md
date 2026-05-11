# scikit-learn (my own notes)

So first of all we need to install all our packages. Four scikit-learn we need: scikit-learn, numpy (for working with arrays) , pandas (for working with dataframes), matplotlib (for visualisation) and jupyterlab. JupyterLab is very useful for Data Science and Machine Learning because it allows code to be executed step by step instead of running the entire program at once. This makes debugging easier, helps analyze data interactively, and allows results such as tables and visualizations to appear directly below the code. Variables and datasets remain in memory, which makes experimentation faster and more efficient, especially when working with pandas, time series, sensor data, and machine learning models.
```python
pip3 install scikit-learn numpy pandas matplotlib 
```
```python
pip3 install jupyterlab
```
Then open the jupyter lab window:

```python
jupyter lab
```
What I am going to do is to train simple classification model to detect breast cancer.
```python
from sklearn.datasets import load_breast_cancer        # loads the data_set
from sklearn.model_selection import train_test_split   # allows me to split the data into a training portion and into a testing portion
from sklearn.preprocessing import StandardScaler       # for scaling the data
from sklearn.neighbors import KNeighborsClassifier     # special type of classifier (breast cancer yes or no)
```
```python
data = load_breast_cancer(as_frame=True).frame         # loaded as a data frame
```
```python
data                                                   # data will show up after this command
```
![Breast cancer properties](images/dataset_breast_cancer.png)
