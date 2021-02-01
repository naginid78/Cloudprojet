from utils import DataHandling, FeatureRecipe, FeatureExtractor, ModelBuilder

def DataManager(dh:DataHandling=None, fr:FeatureRecipe=None, fe:FeatureExtractor=None):
    """
        Function linking the 3 first classes of the pipeline
    """
    # Columns to keep
    klist=['Manufacturer','Latest_Launch','Horsepower','Price_in_thousands']

    dh = DataHandling()
    dh.get_data()
    fr = FeatureRecipe(dh.data)
    fr.prepare_data(0.3)

    fe = FeatureExtractor(fr.data,klist)
    return fe.split_data(0.1)

X_train, X_test, y_train, y_test = DataManager()
m = ModelBuilder() 
m.train(X_train, y_train)
m.print_accuracy(X_test,y_test)
m.save_model('.')
