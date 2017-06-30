#!/usr/bin/env python3

import pandas as pd
import numpy as np
import math
from sklearn.cross_validation import KFold
from xgboost import XGBRegressor

CV_N_FOLDS = 5
SEED = 134


def feat_eng(df, traintest):

    month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    df['month_year_cnt'] = month_year.map(month_year_cnt_map)

    week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    df['week_year_cnt'] = week_year.map(week_year_cnt_map)

    df.drop(["timestamp"], axis=1, inplace=True)

    df["product_type"] = df["product_type"].astype("category")
    df["product_type"].cat.set_categories(["Investment", "OwnerOccupier"], inplace=True)
    df = pd.concat([df, pd.get_dummies(df["product_type"], prefix="product_type")], axis=1)
    df.drop("product_type", axis=1, inplace=True)

    df.loc[df.life_sq.isnull(), "life_sq"] = df[df.life_sq.isnull()].full_sq * 0.75
    df.loc[df.life_sq > df.full_sq, "life_sq"] = df[df.life_sq.isnull()].full_sq * 0.75

    df.loc[df.kitch_sq >= df.life_sq, "kitch_sq"] = np.nan
    df["kitch_sq"] = df.kitch_sq.map(lambda x: x if x > 1 else np.nan)
    df.loc[(df.kitch_sq.isnull()) & (df.life_sq.notnull()), "kitch_sq"] = \
        df[(df.kitch_sq.isnull()) & (df.life_sq.notnull())].life_sq * 0.28

    traintest.loc[traintest.build_year < 1000, "build_year"] = np.nan
    traintest.loc[traintest.build_year > 2020, "build_year"] = np.nan
    df.loc[df.build_year < 1000, "build_year"] = np.nan
    df.loc[df.build_year > 2020, "build_year"] = np.nan

    traintest_slice = traintest[(traintest.sub_area.notnull()) & (traintest.build_year.notnull())]
    avg_build_year_per_sub_area = traintest_slice[["sub_area", "build_year"]].groupby("sub_area").median()
    avg_build_year_per_sub_area["build_year"] = avg_build_year_per_sub_area.build_year.map(int)
    df.loc[df.build_year.isnull(), "build_year"] = \
        df[df.build_year.isnull()].sub_area.map(avg_build_year_per_sub_area.build_year.to_dict())

    traintest_slice = traintest[(traintest.sub_area.notnull()) & (traintest.max_floor.notnull())]
    avg_max_floor_per_sub_area = traintest_slice[["sub_area", "max_floor"]].groupby("sub_area").median()
    avg_max_floor_per_sub_area["max_floor"] = avg_max_floor_per_sub_area.max_floor.map(int)
    df.loc[df.max_floor.isnull(), "max_floor"] = \
        df[df.max_floor.isnull()].sub_area.map(avg_max_floor_per_sub_area.max_floor.to_dict())
    df.loc[df.max_floor < df.floor, "max_floor"] = np.nan

    df.loc[df.state == 33, "state"] = 3

    traintest_slice = traintest[(traintest.sub_area.notnull()) & (traintest.state.notnull())]
    mode_state_per_sub_area = traintest_slice.groupby("sub_area")["state"].agg(lambda x: x.value_counts().index[0])
    df.loc[df.state.isnull(), "state"] = \
        df[df.state.isnull()].sub_area.map(mode_state_per_sub_area.to_dict())

    return df.select_dtypes(include=[np.number])


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5


# is this data science anymore?
def m123(x):
    if abs(x - 1000000) < 250000:
        return 1000000
    if abs(x - 2000000) < 250000:
        return 2000000
    if abs(x - 3000000) < 250000:
        return 3000000
    return x


def cross_validate(model, df):
    scores = list()

    print("# Applying %d-fold cross-validation" % CV_N_FOLDS)
    kf = KFold(len(df), n_folds=CV_N_FOLDS)
    for index, (train, test) in enumerate(kf):
        df_train = df.ix[train]
        df_test = df.ix[test]

        model.fit(
            df_train.drop(["price_doc", "id"], axis=1).values,
            df_train["price_doc"].values)

        score = rmsle(
            df_test["price_doc"].values,
            list(map(m123, model.predict(df_test.drop(["price_doc", "id"], axis=1).values)))
        )
        print("Fold %d score: %0.5f" % (index, score))
        scores.append(score)

    print("Avg score: %0.5f (+/- %0.5f)" % (np.average(scores), np.std(scores)))


def main():
    train = pd.read_csv("data/train.csv", parse_dates=["timestamp"])
    test = pd.read_csv("data/test.csv", parse_dates=["timestamp"])
    traintest = pd.concat([train, test])

    train = feat_eng(train, traintest)
    test = feat_eng(test, traintest)

    val = train[25001:]
    _train = train[:25000]

    model = XGBRegressor(
        seed=SEED,
        silent=True,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        n_estimators=500
    )

    cross_validate(model, _train)

    model.fit(_train.drop(["price_doc"], axis=1).values, _train.price_doc.values)

    val_y = val.price_doc.values
    val_y_pred = model.predict(val.drop(["price_doc"], axis=1).values)
    val_y_pred = map(lambda x: x * 0.95, val_y_pred)
    val_y_pred = map(m123, val_y_pred)
    score = rmsle(
        val_y,
        list(val_y_pred)
    )
    print("Validation score: %0.5f" % score)

    model.fit(train.drop(["price_doc"], axis=1).values, train.price_doc.values)

    test["price_doc"] = model.predict(test.values)
    test["price_doc"] = test.price_doc.map(lambda x: x * 0.95)
    test["price_doc"] = test.price_doc.map(m123)
    test.to_csv('./predictions.csv', columns=["id", "price_doc"], index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
