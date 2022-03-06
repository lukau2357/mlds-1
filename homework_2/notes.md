Nominal independent variables:
    Movement (dribble or cut is reference)
    Competition (EURO is reference)
    PlayerType (C - center is reference)
    Transition (binary, no need for dummy encoding)
    TwoLegged (binary, no need for dummy encoding)

ShotType is the target variable and the reference category is tip-in

    # MLR target variable encoding
    y_encoding = {
        "above head": 0,
        "layup": 1,
        "other": 2,
        "hook shot": 3,
        "dunk": 4,
        "tip-in": 5
    }

    ~ 214 iterations for training MLR with LBFGS on the full dataset
    ~ 75% precision for MLR (with intercepts) taking a naive 80-20 evaluatuion on the given dataset (0.7522388059701492)
