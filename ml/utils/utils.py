import pandas as pd
import numpy as np

class DataHandling:
    """
        Getting data from local csv file
    """
    def __init__(self):
        """
            Initializing the dataset handling
        """
        print("DataHandling intialization")
        self.data = None
        print("Intialization done !")

    def get_data(self):
        print("Loading data from local file...")
        self.data=pd.read_csv("..\data\Car_sales.csv")
        print("Dataset shape : {} lines, {} columns".format(self.data.shape[0],self.data.shape[1]))
        print("Data loaded from local file !")

class FeatureRecipe:
    """
        Cleaning the dataset
    """
    def __init__(self,data:pd.DataFrame):
        """
            Initialazing the FeatureRecipe
        """
        print("FeatureRecipe intialization...")
        self.data = data
        self.categorial = []
        self.continuous = []
        self.discrete = []
        self.drop = []
        print("Initialization done !")

    def convert_and_encode(self):
        def getrange(Price_in_thousands):
            if (Price_in_thousands >= 0 and Price_in_thousands < 15):
                return '0 - 15'
            if (Price_in_thousands >= 15 and Price_in_thousands < 30):
                return '15 - 30'
            if (Price_in_thousands >= 30 and Price_in_thousands < 45):
                return '30 - 45'
        self.data['echelle_de_prix'] = self.data.apply(lambda x:getrange(x['Price_in_thousands']),axis = 1)
        def getrangeh(Horsepower):
            if (Horsepower >= 92 and Horsepower < 127):
                return '92 - 127'
            if (Horsepower >= 127 and Horsepower < 162):
                return '127 - 162'
            if (Horsepower >= 162 and Horsepower < 197):
                return '162 - 197'
            if (Horsepower >= 197 and Horsepower < 276):
                return '197 - 276'
        self.data['echelle_de_chevaux'] = self.data.apply(lambda x:getrangeh(x['Horsepower']),axis = 1)


    def separate_variable_types(self) -> None:
        """
            Separating dataset features by type
        """
        print("Separating features by type...")
        for col in self.data.columns:
            if self.data[col].dtypes == 'int64':
                self.discrete.append(self.data[col])
            elif self.data[col].dtypes == 'float64':
                self.continuous.append(self.data[col])
            else:
                self.categorial.append(self.data[col])
        print ("Dataset columns : {} \nNumber of discrete features : {} \nNumber of continuous features : {} \nNumber of categorical features : {} \nTotal number of features : {}\n".format(len(self.data.columns),len(self.discrete),len(self.continuous),len(self.categorial),len(self.discrete)+len(self.continuous)+len(self.categorial) ))

    def drop_useless_features(self) :
        """
            Droping useless features and observations
        """
        print("Dropping useless features and observations...")
        # Dropping observations with price == 0
        self.data.drop(self.data[self.data['Price_in_thousands'] == 0].index, inplace=True)
        # Dropping duplicates null features observations
        self.data.dropna(inplace=True,axis=0)
        # Dropping extreme observations
        self.data.drop(self.data[self.data['Horsepower'] >= 305].index,inplace=True)
        self.data.drop(self.data[self.data['Sales_in_thousands'] >= 150].index,inplace=True)
        self.data.drop(self.data[self.data['Price_in_thousands'] >= 45].index,inplace=True)
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        self.data = self.data[~((self.data < (Q1 - 1.5*IQR))|(self.data > (Q3 + 1.5*IQR))).any(axis = 1)]
        print("Useless features and observations dropped !\n")

    def drop_duplicate(self) :
        """
            Dropping duplicates features
        """
        def get_duplicate(data:pd.DataFrame): 
              # Create an empty set 
            duplicateColumnNames = set() 
      
            # Iterate through all the columns  
            # of dataframe 
            for x in range(data.shape[1]): 
          
                # Take column at xth index. 
                col = data.iloc[:, x] 
          
                # Iterate through all the columns in 
                # DataFrame from (x + 1)th index to 
                # last index 
                for y in range(x + 1, data.shape[1]): 
              
                    # Take column at yth index. 
                    otherCol = data.iloc[:, y] 
              
                    # Check if two columns at x & y 
                    # index are equal or not, 
                    # if equal then adding  
                    # to the set 
                    if col.equals(otherCol): 
                            duplicateColumnNames.add(data.columns.values[y]) 
                  
            # Return list of unique column names  
            # whose contents are duplicates. 
            return list(duplicateColumnNames) 
        self.data = self.data.drop(get_duplicate(self.data), axis=1)
   
    def drop_na_prct(self,threshold : float):
        """
            Threshold between 0 and 1.
            Dropping features with NaN ratio > threshold
            Counting dropped features
            params: threshold : float
        """
        count = 0
        print("Dropping features with more than {} % NaN...".format(threshold*100))
        for col in self.data.columns:
            if self.data[col].isna().sum()/self.data.shape[0] >= threshold:
                self.drop.append( self.data.drop([col], axis='columns', inplace=True) )
                count+=1
        print("Dropped {} features.\n".format(count))

    def prepare_data(self,threshold : float):
        self.drop_useless_features()
        self.drop_na_prct(threshold)
        self.drop_duplicate()
        self.separate_variable_types()
        self.convert_and_encode()
        print("Processed dataset shape : {} lines, {} columns".format(self.data.shape[0],self.data.shape[1]))
        print("FeatureRecipe processing done !\n")

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

class FeatureExtractor:
    """
    Feature Extractor class
    """
    def __init__(self, data: pd.DataFrame, flist: list):
        """
            Input : pandas.DataFrame, feature list to select
            Output : X_train, X_test, y_train, y_test according to sklearn.model_selection.train_test_split
        """
        print("FeatureExtractor initialization...")
        self.X_train, self.X_test, self.y_train, self.y_test = None,None,None,None
        self.data = data
        self.flist = flist
        print("Intialisation done !\n")

    def extractor(self):
        print("Extracting selected columns...")
        for col in self.flist:
            if col in self.data:
                self.data.drop(col, axis=1, inplace=True)
        print("Selected columns extracted ! \n")

    def splitting(self, size:float,rng:int, y:str):
        print("splitting dataset for train and test")
        x = self.data.loc[:,self.data.columns != y]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, self.data[y], test_size=size, random_state=rng)
        print("splitting done")

    def extract_data(self):
        self.extractor()
        self.splitting(0.3,42,'FEATURE')
        print("done processing Feature Extractor")
        return self.X_train, self.X_test, self.y_train, self.y_test
    def split_data(self,split:float):
        self.extractor()
        self.splitting(split,42,'Price_in_thousands')
        print("FeatureExtractor processing done !\n")
        return self.X_train, self.X_test, self.y_train, self.y_test



class ModelBuilder:
    """
        Training and printing machine learning model
    """
    def __init__(self, model_path: str = None, save: bool = None):

        print("ModelBuilder initialization...")
        self.model_path = model_path
        self.save = save
        self.line_reg = LinearRegression()
        print ("Initialization done !")

    def train(self, X, y):
        self.line_reg.fit(X,y)

    def __repr__(self):
        pass

    def predict_test(self, X) -> np.ndarray:
        # on test sur une ligne
        return self.line_reg.predict(X)

    def save_model(self, path:str):

        #with the format : 'model_{}_{}'.format(date)
        #joblib.dump(self, path, 3)
        pass

    def predict_from_dump(self, X) -> np.ndarray:
        pass

    def print_accuracy(self,X_test,y_test):
        self.line_reg.predict(X_test)
        self.line_reg.score(X_test,y_test)*100

    def load_model(self):
        try:
            #load model
            pass
        except:
            pass

