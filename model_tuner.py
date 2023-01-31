# This is the class for training the classifier (not)/profitable. It requires the raw_feature.csv to be created by feature_eng.ipynb (?)

import pandas as pd
import numpy as np
import random
import datetime

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, recall_score, roc_curve, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

class ModelTuner(dict):

    def __init__(self, **kwargs) -> None:
        self.__dict__ = kwargs
        self.run_pipeline()
        
    def run_pipeline(self):
        print(f"=========== {str(datetime.datetime.now())[:-7]} ===========")
        self.import_features()
        self.split_data(self.split_method)
        self.build_classifiers(self.classifier_params, self.classifier_seeds)
        self.train_classifiers()
        self.predict(self.threshold)
        self.evaluate()

    ### FEATURE HANDLING ###

    def import_features(self) -> None:
        # print("Loading features file...")
        self.F = pd.read_csv(self.features_file)

    def get_features_df(self) -> pd.DataFrame:
        """
            Returns a copy of the features dataframe
        """
        return self.F.copy()

    ### SPLITTING ###

    def split_data(self, method: str) -> None:
        """
            Separate X,y, and INFO.
            Split features dataframe with one of the 3 methods:
            * random: requires split_seed and test_size_pct
            * (by) date: requires test_split_date
            * (by) ticker: requires test_tickers (list)
        """

        # print("Separating X/y/INFO")

        self.X    = self.F[self.columns_x]
        self.y    = self.F[self.columns_y]
        self.INFO = self.F[self.columns_info]

        if method=='random':
            self.split_random(self.split_seed, self.test_size_pct)
        elif method=='date':
            self.split_by_date(self.test_split_date)
        elif method=='ticker':
            self.split_by_ticker(self.test_tickers)
        else:
            Exception(f'Unknown split method: {method}')

        self.X_train,    self.X_test    = self.X.iloc[self.ndx_train],    self.X.iloc[self.ndx_test]
        self.y_train,    self.y_test    = self.y.iloc[self.ndx_train],    self.y.iloc[self.ndx_test]
        self.INFO_train, self.INFO_test = self.INFO.iloc[self.ndx_train], self.INFO.iloc[self.ndx_test]

        self.summarize_split()

    def split_random(self, seed: int, test_pct: float) -> None:
        """
            Splitting randomly
        """

        print(f"Splitting randomly (seed={seed})")

        all_indices = self.INFO.index.to_list()
        random.Random(seed).shuffle(all_indices)
        
        N = len(all_indices)
        N_train = N - int(N*test_pct)
        
        self.ndx_train = all_indices[:N_train]
        self.ndx_test  = all_indices[N_train:]

    def split_by_date(self, split_date: str) -> None:
        """
            Splitting features df by date. Training is prior to split_date and testing posterior.
        """

        print(f"Splitting by date (split_date={split_date})")

        self.ndx_train = self.INFO[self.INFO.earnings_date< split_date].index.to_list()
        self.ndx_test  = self.INFO[self.INFO.earnings_date>=split_date].index.to_list()

        random.shuffle(self.ndx_train)
        random.shuffle(self.ndx_test)

    def split_by_ticker(self, test_tickers: list) -> None:
        """
            Splitting features df by tickers. Selected tickers are only present in testing.
        """

        print(f"Splitting by ticker (test_tickers={test_tickers})")

        all_indices = self.INFO.index.to_list()
        random.shuffle(all_indices)

        self.ndx_train = [ndx for ndx in all_indices if self.INFO.iloc[ndx]['ticker'] not in test_tickers]
        self.ndx_test  = [ndx for ndx in all_indices if self.INFO.iloc[ndx]['ticker']     in test_tickers]

    def summarize_split(self) -> None:
        print("Split summary:")
        print(f"  shapes: X {self.X.shape}, y {self.y.shape}, INFO {self.INFO.shape}")
        print(f"  training size: {self.X_train.shape[0]}, testing size: {self.X_test.shape[0]}, test: {round(100.0*self.X_test.shape[0]/self.X.shape[0],1)}%")

    ### CLASSIFIER ###

    def build_classifiers(self, classifier_params: dict, seeds: list) -> None:
        # print(f"Building classifiers with {len(seeds)} different seeds...")
        self.clfs = [GradientBoostingClassifier(**classifier_params, random_state=seed) for seed in seeds]
        # print(f"Classifier params: {self.clfs[0].get_params()}")

    def train_classifiers(self) -> None:
        print("Training classifiers.",end='')
        for i,_ in enumerate(self.clfs):
            self.clfs[i].fit(self.X_train, self.y_train[self.columns_y[0]])
            print(".",end='')
        print(" complete.")

    def predict(self, threshold) -> None:
        # print("Predicting class of tradeability...")
        self.y_preds = []
        for i,_ in enumerate(self.clfs):
            y_pred_proba = self.clfs[i].predict_proba(self.X_test)
            y_pred = (y_pred_proba[:,1]>=threshold).astype(int)
            self.y_preds.append(y_pred)

    ### EVALUATION ###

    def evaluate(self) -> None:
        """
            Note: decimal scores are presented as %
        """
        print("EVALUATION (~avg score, invididual scores)")

        ## PRECISION - how many predictions were right?##
        P = []
        for y_pred in self.y_preds:
            p  = precision_score(self.y_test.to_numpy()[:,0], y_pred)
            P.append(p * 100.0)
        print(f"  Precision: {round(np.mean(P))}% ({','.join([str(round(p)) for p in P])})")
        print(f"        std: {round(np.std(P))}%")
        avg_train_precision = self.evaluate_training()
        print(f"  (Train P): {round(100*avg_train_precision)}%")
        

        ## RECALL - how many actual good hits we found? % ##
        R = []
        for y_pred in self.y_preds:
            r  = recall_score(self.y_test.to_numpy()[:,0], y_pred)
            R.append(r * 100.0)
        print(f"  Recall:    {round(np.mean(R))}% ({','.join([str(round(r)) for r in R])})")

        ## QUANTITY  - how many trade occasions have been found? ##
        T = []
        for y_pred in self.y_preds:
            cm  = confusion_matrix(self.y_test.to_numpy()[:,0], y_pred)
            T.append( cm[:,1].sum() )
        print(f"  Trades:    {round(np.mean(T))}  ({','.join([str(t) for t in T])})")

        ## PREP OUTPUT ##
        self.out = {
            'precision': np.mean(P),
            'recall':    np.mean(R)
        }

    def evaluate_training(self) -> float:
        P =[]
        for i,_ in enumerate(self.clfs):
            y_pred_proba = self.clfs[i].predict_proba(self.X_train)
            y_pred = (y_pred_proba[:,1]>=self.threshold).astype(int)
            p  = precision_score(self.y_train.to_numpy()[:,0], y_pred)
            P.append(p)
        return np.mean(P)

    def feature_importance(self) -> None:
        """
            Avg feature importances
            to know which to remove...
        """
        f_imp = None
        for i,_ in enumerate(self.clfs):
            f_imp_i = pd.DataFrame({
                    'feature':self.X.columns.to_list(),
                    f'imp_{i}':self.clfs[i].feature_importances_})\
                    .set_index('feature')
            if f_imp is None:
                f_imp = f_imp_i
            else:
                f_imp = f_imp.join(f_imp_i)
        f_imp = f_imp.mean(axis=1).sort_values(ascending=False)
        print(f_imp)

    ## PERSISTING TO SIMULATOR ##

    def persist_artifacts(self, location: str) -> None:
        out = self.F.iloc[self.ndx_test]
        out.loc[:,'pred'] = self.aggregate_predictions()['mean'].to_numpy()
        out.to_csv(location, index=False)

    def aggregate_predictions(self) -> pd.DataFrame:
        """
            This is to make sure the simulator can learn on test sets
        """
        agg = pd.DataFrame()
        for i,_ in enumerate(self.clfs):
            y_proba = self.clfs[i].predict_proba(self.X_test)[:,1]
            agg[str(i)] = y_proba
        agg.loc[:,'mean'] = agg.mean(axis=1)
        agg.to_csv('tmp.csv')
        return agg