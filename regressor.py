import os
import pickle
import numpy as np
import pandas as pd
import hyperopt
import sklearn

from typing import Optional
from functools import partial
from collections import defaultdict
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, Trials, STATUS_OK

"""
This class implementation was based on 
https://github.com/stasulam/tmle/tree/master/tmle
repository.

It was changed in order to suit regression problems
"""

class RegressorOptimizer(object):

    def __init__(
            self,
            regressor: sklearn.base.RegressorMixin,
            space: dict,
            metric: sklearn.metrics
    ) -> None:
        """RegressorOptimizer.
        :param regressor: regressor.
        :param space: space from which the hyperparameters values are drawn.
        :param metric: the metric used to assess the model's performance.
        """
        self.regressor = regressor
        self.space = space
        self.metric = metric

    def find_best_params(
            self,
            X: np.ndarray,
            y: np.ndarray,
            experiments_path: str,
            experiments_name: str,
            max_evals: int = 10,
            n_splits: int = 3,
            overfit_penalty: Optional[float] = None,
            verbose: bool = True,
            **params
    ) -> Optional[dict]:
        """Find hyperparameters that minimize user-defined metric (`self.metric`).
        This method uses the Tree Parzen Estimator [1] to find hyperparameters values
        that minimize the user-defined loss function (although, at this stage of development
        it supports only metrics that use the output of `predict` method).
        Obtained results are then stored in the `Trials` object and saved in given directory.
        Experiment can be interrupted - results will be saved on exit.
        References:
        [1] Bergstra, James S., et al. “Algorithms for hyper-parameter optimization.”
        Advances in Neural Information Processing Systems. 2011.
        :param X: array-like or spare matrix of shape = [n_samples, n_features].
        :param y: array-like, shape = [n_samples] or [n_samples, n_outputs].
        :param experiments_path: path to directory when experiments are stored.
        :param experiments_name: name of given experiment.
        :param max_evals: allow up to this many function evaluations before returning.
        :param n_splits: number of splits used during cross-validation.
        :param overfit_penalty: additional penalty for overfitting (at training stage).
        :param verbose: print out some information to stdout during search.
        :return: dictionary with best parameters for given experiment run.
        """
        if os.path.exists(os.path.join(experiments_path, '.'.join([experiments_name, 'hpopt']))):
            trials = pickle.load(open(os.path.join(experiments_path, '.'.join([experiments_name, 'hpopt'])), 'rb'))
            max_evals = len(trials.trials) + max_evals
        else:
            trials = Trials()

        try:
            best_params = fmin(
                fn=partial(
                    self.evaluate_params,
                    X=X, y=y,
                    n_splits=n_splits,
                    overfit_penalty=overfit_penalty,
                    verbose=verbose,
                    **params
                ),
                space=self.space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials
            )
        except KeyboardInterrupt:
            # saving experiments on exit
            self._save_experiments_results(
                trials=trials,
                experiments_path=experiments_path,
                experiments_name=experiments_name
            )
            return None

        # save trials object for further usage
        self._save_experiments_results(
            trials=trials,
            experiments_path=experiments_path,
            experiments_name=experiments_name
        )

        return best_params

    def evaluate_params(
            self,
            clf_params: dict,
            X: np.ndarray,
            y: np.ndarray,
            n_splits: int = 3,
            overfit_penalty: Optional[float] = None,
            verbose: bool = True,
            **params
    ) -> dict:
        """Evaluate hyperparameters.
        This method was designed to evaluate a given set of parameters in conditions
        of continous dataset (hence, `KFold` was used). Furthermore, a decision
        was made to further penalize the overfitting (`overfit_penalty` simply adds some
        constant to loss function in order to discourage `TPE` from sampling similar sets
        of hyperparameters.
        :param X: array-like or spare matrix of shape = [n_samples, n_features].
        :param y: array-like, shape = [n_samples] or [n_samples, n_outputs].
        :param clf_params: hyperparameters passed to given regressor.
        :param n_splits: number of splits used during cross-validation.
        :param overfit_penalty: additional penalty for overfitting (at training stage).
        :param verbose: print out some information for given experiment run.
        :return: dictionary with loss and information about metric values.
        """
        self.regressor.set_params(**clf_params)
        score_train, score_valid = [], []
        for train_idx, valid_idx in KFold(n_splits=n_splits).split(X, y):
            x_train_fold, x_valid_fold = X[train_idx], X[valid_idx]
            y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]
            self.regressor.fit(x_train_fold, y_train_fold, **params)
            score_train.append(self.metric(y_train_fold, self.regressor.predict(x_train_fold)))
            score_valid.append(self.metric(y_valid_fold, self.regressor.predict(x_valid_fold)))
        mean_score_train = np.mean(score_train)
        mean_score_valid = np.mean(score_valid)
        if verbose:
            msg = 'Train: {score_train:.4f}, valid: {score_valid:.4f}'
            print(msg.format(score_train=mean_score_train, score_valid=mean_score_valid))
        loss = mean_score_valid
        if overfit_penalty:
            loss += np.where(mean_score_valid/mean_score_train - 1 > overfit_penalty, abs(mean_score_valid-mean_score_train)/2, 0)
        return {
            'loss': loss,
            'status': STATUS_OK,
            'score': {'train': mean_score_train, 'valid': mean_score_valid}
        }

    def evaluate_experiments(self, trials: hyperopt.Trials.trials):
        """Evaluate experiments conducted during training stage.
        :param trials: object that stores information about experiments.
        :return: DataFrame with summary of experiments.
        """
        experiments = defaultdict(list)
        for trial in trials:
            for param, value in self.space_eval(trial).items():
                experiments[param].append(value)
            for dataset, score in trial['result']['score'].items():
                experiments[dataset].append(score)
        return pd.DataFrame.from_dict(experiments)

    def space_eval(self, trials: hyperopt.Trials.trials) -> dict:
        """Evaluate hyperparameters space and return best params as dict.
        `space_eval' from hyperopt is broken. See:
            https://github.com/hyperopt/hyperopt/issues/383
        :param trials: object that stores information about experiments.
        :return: dictionary with best parameters.
        """
        params = dict()
        for param in trials['misc']['vals']:
            params[param] = trials['misc']['vals'][param][0]
        return hyperopt.space_eval(self.space, params)

    def _save_experiments_results(
            self,
            trials: hyperopt.Trials,
            experiments_path: str,
            experiments_name: str
    ) -> None:
        """Save experiments results.
        :param trials: object that stores information about experiments.
        :param experiments_path: path to directory when experiments are stored.
        :param experiments_name: name of given experiment.
        """
        if not os.path.exists(experiments_path):
            os.makedirs(experiments_path)
        with open(os.path.join(experiments_path, '.'.join([experiments_name, 'hpopt'])), 'wb') as experiments:
            pickle.dump(trials, experiments)