
import pandas as pd
import pytest

from src.features.build_features import (
    create_average_score,
    create_score_trend,
    create_recent_score_change,
    create_score_volatility
)


def test_feature_engineering_functions():

    df = pd.DataFrame({
        "previous_score_1": [45],
        "previous_score_2": [48],
        "previous_score_3": [50]
    })

    df = create_average_score(df)

    df = create_score_trend(df)

    df = create_recent_score_change(df)

    df = create_score_volatility(df)

    assert df["average_previous_score"].iloc[0] == pytest.approx(
        47.6666667
    )

    assert df["score_trend"].iloc[0] == -2.5

    assert df["recent_score_change"].iloc[0] == -3

    assert df["score_volatility"].iloc[0] == pytest.approx(
        2.0548047
    )
