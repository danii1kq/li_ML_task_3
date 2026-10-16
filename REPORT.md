Model card: term deposit telemarketing

Purpose

The call centre can reach only 10% of the clients on a list per campaign wave. This model ranks a client list so that the top 10%, the clients who get called, are the ones most likely to say yes to a term deposit. Scoring happens before any call and uses only information that exists at that moment. The model orders the list; eligibility rules stay with the bank.

Data

UCI Bank Marketing (Moro, Rita and Cortez, 2014), as prepared by the course: data/train.csv with 32,940 labelled rows (11.3% yes) and data/test.csv with 8,236 rows to score. Nineteen features are used. The answer column y exists only in train, and duration was excluded on purpose: it is recorded after the call and has the largest correlation with the answer in the file (0.41), so it would not exist when the list is scored.

Two quirks of the data are handled in the pipeline. The value unknown appears in six categorical columns (10,199 cells; 6,871 of them in default), and it is kept as its own category because it behaves like one: clients with default = unknown accept at 5.2%, against 12.9% for default = no. The value pdays = 999 (31,694 rows, 96.2%) is a code meaning not contacted before, so it becomes a flag column and the original value is median-imputed. previous = 0 and poutcome = nonexistent mark the same 28,446 clients, the never-contacted group.

The rows in the file are shuffled (id simply runs from 0 to 32,939), so the campaign periods are mixed, and my own split of the training data is random as well. SEED = 2175 (last four digits of my student ID) is used in every split and every random model.

Pipeline

Everything that learns from data sits inside one scikit-learn Pipeline, so cross-validation fits encoders and scalers on training folds only. Numeric columns go through median imputation and standard scaling; pdays gets its own branch with the flag. Categorical columns are one-hot encoded, and unseen values are tolerated. The final estimator is a random forest (300 trees, max_depth=16, min_samples_leaf=5, random_state=2175).

It was picked on 5 stratified folds of the 80% training part, against three other families: logistic regression, KNN and a decision tree, each tuned by grid search. The forest led on average precision (0.4686 against 0.4507 for logistic regression); the gap is about one fold spread, so this is a mild preference, not a proven win. Class weights did not improve any metric, so the models stay unweighted. The comparison tables are in li_task3.ipynb.

Metric and results

This is a ranking task under a fixed budget, so the metrics are ROC-AUC and average precision, plus the business numbers at the 10% cut. Accuracy is not used: the majority class alone gives 88.7%.

On my hold-out (6,588 clients, 742 yes) the model scores ROC-AUC 0.8071 and average precision 0.4825. The 10% rule calls 659 clients; 350 of them say yes (precision 0.531), 350 of the 742 yes clients are reached (recall 0.472) and 392 are missed. Calling 659 clients at random would reach about 74 yes clients, so the lift is 4.7x. Of the 659 calls, 309 are wasted.

For submission the pipeline was refit on all 32,940 training rows and used to score data/test.csv: 824 of 8,236 clients (10.0%) are marked for a call. model.joblib (37.9 MB) loads in a fresh Python process and scores the raw test.csv columns; predictions.csv holds the columns id, y_pred and y_score.

Limits and ways it can fail

The random split flatters the result. Train and hold-out share the same campaign periods and the same economy. Real use means training on the past and calling into the future, with different months and different euribor3m levels, so 0.8071 is an upper bound; expect the live quality to be lower.

The 10% cut is mechanical. The model calls the top decile of whatever list it is given. If a future list or a new economic period shifts the scores, precision at the top falls below 0.53 with no signal from the score itself.

The meaning of the columns can drift. unknown means something different in each column, and if the bank changes how it records clients, the same categories start to mean something else while the model keeps producing confident scores.

The list is better than random and far from perfect: about half of the called clients still say no (precision 0.531), and more than half of the eventual buyers are never called (recall 0.472).

Never add duration back into the pipeline. It would make every offline number look better and it does not exist at scoring time.

Monitoring

After launch I would track, per campaign wave: the realized yes rate among the called clients (against the 0.53 baseline), the share of yes clients that ended up in the called decile, and the distribution of the y_score values on incoming lists. The economic columns and the yes base rate are natural early warnings. If top-decile precision falls clearly below the hold-out level, retrain on the newest campaigns and refit the threshold. Versions are pinned in requirements.txt (scikit-learn 1.7.2), and li_task3.ipynb reproduces everything top to bottom with SEED = 2175.