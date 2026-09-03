"""
BACKWARD-COMPATIBILITY SHIM — do not add new code here.

cards_model.py was consolidated into cards.py. This shim exists solely so
that card_model.pkl (trained and pickled against the OLD module path, e.g.
`cards_model.CardPredictionModel`) can still be unpickled by joblib without
retraining. joblib/pickle resolve a saved object's class by its original
module string, so this module must keep existing under this name.

New code should `import cards`, not this module. If you retrain the card
models (train.py / update_models.py), the newly written card_model.pkl will
be pickled against `cards`, and this shim can eventually be deleted once no
old artifact depends on it anymore.
"""
from cards import *  # noqa: F401,F403
from cards import (
    CardPredictionModel, CARD_SCHEMA_VERSION, build_card_features,
    train_card_models, predict_cards,
)
