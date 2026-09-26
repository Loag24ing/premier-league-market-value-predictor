from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Premier League Market Value Predictor",
    page_icon="⚽",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
GLOBAL_MODEL_PATH = BASE_DIR / "models" / "final_global_model.pkl"
DEFENDER_MODEL_PATH = BASE_DIR / "models" / "final_defender_model.pkl"

GLOBAL_FEATURES = [
    "position",
    "age",
    "age_squared",
    "gamesPlayed",
    "starts",
    "timePlayed",
    "goals_per90",
    "goalAssists_per90",
    "totalShots_per90",
    "expectedGoals_per90",
    "expectedAssists_per90",
    "keyPassesAttemptAssists_per90",
    "successfulDribbles_per90",
    "totalTackles_per90",
    "tacklesWon_per90",
    "interceptions_per90",
    "blocks_per90",
    "totalClearances_per90",
    "duelsWon_per90",
    "groundDuelsWon_per90",
    "aerialDuelsWon_per90",
    "recoveries_per90",
    "totalLossesOfPossession_per90",
    "totalTouchesInOppositionBox_per90",
]

DEFENDER_FEATURES = [
    "age",
    "age_squared",
    "gamesPlayed",
    "starts",
    "timePlayed",
    "totalTackles_per90",
    "tacklesWon_per90",
    "tackle_win_rate",
    "interceptions_per90",
    "blocks_per90",
    "totalClearances_per90",
    "duelsWon_per90",
    "groundDuelsWon_per90",
    "aerialDuels_per90",
    "aerialDuelsWon_per90",
    "aerial_win_rate",
    "recoveries_per90",
    "totalPasses_per90",
    "successfulPassesOwnHalf_per90",
    "successfulPassesOppositionHalf_per90",
    "successfulLongPasses_per90",
    "openPlayPasses_per90",
    "touches_per90",
    "own_half_pass_share",
    "opp_half_pass_share",
    "long_pass_share",
    "goalAssists_per90",
    "expectedAssists_per90",
    "keyPassesAttemptAssists_per90",
    "totalLossesOfPossession_per90",
]


@st.cache_resource
def load_models():
    global_model = joblib.load(GLOBAL_MODEL_PATH)
    defender_model = joblib.load(DEFENDER_MODEL_PATH)
    return global_model, defender_model


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def num(label, value=0.0, min_value=0.0, max_value=None, step=0.1, help_text=None, key=None):
    kwargs = {
        "label": label,
        "value": value,
        "min_value": min_value,
        "step": step,
        "help": help_text,
        "key": key,
    }
    if max_value is not None:
        kwargs["max_value"] = max_value
    return st.number_input(**kwargs)


try:
    global_model, defender_model = load_models()
except Exception as exc:
    st.error("The trained model files could not be loaded.")
    st.code(str(exc))
    st.stop()


st.title("⚽ Premier League Player Market Value Predictor")
st.caption(
    "Estimate a player's market value from Premier League performance metrics using the final hybrid XGBoost model."
)

with st.sidebar:
    st.header("About the model")
    st.markdown(
        """
        **Final architecture**
        - Defenders → defender-specific XGBoost
        - Forwards, midfielders and goalkeepers → global XGBoost

        **2025–26 evaluation**
        - MAE: **€10.85M**
        - RMSE: **€16.56M**
        - R²: **0.537**

        Values are model estimates, not transfer fees or professional scouting valuations.
        """
    )

st.info(
    "Enter season-level per-90 statistics. The training data used players with at least 900 minutes, so predictions are most meaningful for players with substantial playing time."
)

basic_tab, attacking_tab, defensive_tab, passing_tab = st.tabs(
    ["Player & playing time", "Attacking", "Defensive", "Passing (defenders)"]
)

with basic_tab:
    c1, c2 = st.columns(2)
    with c1:
        position = st.selectbox(
            "Position",
            ["Forward", "Midfielder", "Defender", "Goalkeeper"],
        )
        age = num("Age", value=24.0, min_value=16.0, max_value=45.0, step=1.0)
        games_played = num("Games played", value=28.0, min_value=0.0, max_value=38.0, step=1.0)
    with c2:
        starts = num("Starts", value=24.0, min_value=0.0, max_value=38.0, step=1.0)
        time_played = num(
            "Minutes played",
            value=2200.0,
            min_value=0.0,
            max_value=4000.0,
            step=10.0,
            help_text="The model was trained on player-seasons with at least 900 minutes.",
        )

with attacking_tab:
    c1, c2, c3 = st.columns(3)
    with c1:
        goals = num("Goals / 90", 0.20)
        assists = num("Assists / 90", 0.15)
        shots = num("Shots / 90", 1.50)
    with c2:
        xg = num("Expected goals (xG) / 90", 0.20)
        xa = num("Expected assists (xA) / 90", 0.15)
        key_passes = num("Key passes / 90", 1.00)
    with c3:
        dribbles = num("Successful dribbles / 90", 0.80)
        opp_box_touches = num("Touches in opposition box / 90", 3.00)

with defensive_tab:
    c1, c2, c3 = st.columns(3)
    with c1:
        tackles = num("Tackles / 90", 1.50)
        tackles_won = num("Tackles won / 90", 0.90)
        interceptions = num("Interceptions / 90", 1.00)
        blocks = num("Blocks / 90", 0.50)
    with c2:
        clearances = num("Clearances / 90", 2.00)
        duels_won = num("Duels won / 90", 4.00)
        ground_duels_won = num("Ground duels won / 90", 2.50)
        recoveries = num("Recoveries / 90", 5.00)
    with c3:
        aerial_duels = num("Aerial duels / 90", 2.00)
        aerial_duels_won = num("Aerial duels won / 90", 1.20)
        possession_losses = num("Possession losses / 90", 10.00)

with passing_tab:
    if position == "Defender":
        st.write("These features are used by the defender-specific model.")
        c1, c2, c3 = st.columns(3)
        with c1:
            total_passes = num("Total passes / 90", 45.0)
            successful_own_half = num("Successful passes in own half / 90", 20.0)
        with c2:
            successful_opp_half = num("Successful passes in opposition half / 90", 15.0)
            successful_long_passes = num("Successful long passes / 90", 3.0)
        with c3:
            open_play_passes = num("Open-play passes / 90", 40.0)
            touches = num("Touches / 90", 60.0)
    else:
        st.caption("The final global model does not require these extra passing inputs for this position.")
        total_passes = 0.0
        successful_own_half = 0.0
        successful_opp_half = 0.0
        successful_long_passes = 0.0
        open_play_passes = 0.0
        touches = 0.0


st.divider()

if st.button("Predict market value", type="primary", use_container_width=True):
    if time_played < 900:
        st.warning(
            "This player has fewer than 900 minutes. The model was trained after filtering to at least 900 minutes, so this prediction is outside the main training population."
        )

    common = {
        "age": float(age),
        "age_squared": float(age) ** 2,
        "gamesPlayed": float(games_played),
        "starts": float(starts),
        "timePlayed": float(time_played),
        "goals_per90": float(goals),
        "goalAssists_per90": float(assists),
        "totalShots_per90": float(shots),
        "expectedGoals_per90": float(xg),
        "expectedAssists_per90": float(xa),
        "keyPassesAttemptAssists_per90": float(key_passes),
        "successfulDribbles_per90": float(dribbles),
        "totalTackles_per90": float(tackles),
        "tacklesWon_per90": float(tackles_won),
        "interceptions_per90": float(interceptions),
        "blocks_per90": float(blocks),
        "totalClearances_per90": float(clearances),
        "duelsWon_per90": float(duels_won),
        "groundDuelsWon_per90": float(ground_duels_won),
        "aerialDuelsWon_per90": float(aerial_duels_won),
        "recoveries_per90": float(recoveries),
        "totalLossesOfPossession_per90": float(possession_losses),
        "totalTouchesInOppositionBox_per90": float(opp_box_touches),
    }

    try:
        if position == "Defender":
            row = {
                "age": common["age"],
                "age_squared": common["age_squared"],
                "gamesPlayed": common["gamesPlayed"],
                "starts": common["starts"],
                "timePlayed": common["timePlayed"],
                "totalTackles_per90": common["totalTackles_per90"],
                "tacklesWon_per90": common["tacklesWon_per90"],
                "tackle_win_rate": ratio(tackles_won, tackles),
                "interceptions_per90": common["interceptions_per90"],
                "blocks_per90": common["blocks_per90"],
                "totalClearances_per90": common["totalClearances_per90"],
                "duelsWon_per90": common["duelsWon_per90"],
                "groundDuelsWon_per90": common["groundDuelsWon_per90"],
                "aerialDuels_per90": float(aerial_duels),
                "aerialDuelsWon_per90": common["aerialDuelsWon_per90"],
                "aerial_win_rate": ratio(aerial_duels_won, aerial_duels),
                "recoveries_per90": common["recoveries_per90"],
                "totalPasses_per90": float(total_passes),
                "successfulPassesOwnHalf_per90": float(successful_own_half),
                "successfulPassesOppositionHalf_per90": float(successful_opp_half),
                "successfulLongPasses_per90": float(successful_long_passes),
                "openPlayPasses_per90": float(open_play_passes),
                "touches_per90": float(touches),
                "own_half_pass_share": ratio(successful_own_half, total_passes),
                "opp_half_pass_share": ratio(successful_opp_half, total_passes),
                "long_pass_share": ratio(successful_long_passes, total_passes),
                "goalAssists_per90": common["goalAssists_per90"],
                "expectedAssists_per90": common["expectedAssists_per90"],
                "keyPassesAttemptAssists_per90": common["keyPassesAttemptAssists_per90"],
                "totalLossesOfPossession_per90": common["totalLossesOfPossession_per90"],
            }
            input_df = pd.DataFrame([row], columns=DEFENDER_FEATURES)
            prediction = float(defender_model.predict(input_df)[0])
            model_used = "Defender-specific XGBoost"
        else:
            row = {"position": position, **common}
            input_df = pd.DataFrame([row], columns=GLOBAL_FEATURES)
            prediction = float(global_model.predict(input_df)[0])
            model_used = "Global XGBoost"

        prediction = max(0.0, prediction)

        st.success(f"Estimated market value: €{prediction:,.1f}M")
        c1, c2 = st.columns(2)
        c1.metric("Predicted value", f"€{prediction:,.1f}M")
        c2.metric("Model used", model_used)

        with st.expander("Model interpretation note"):
            st.write(
                "The project found systematic underprediction among elite €50M+ players. Market values also reflect reputation, contract length, club bargaining power, international status, scarcity and transfer demand—factors that are not fully captured by the performance inputs."
            )
    except Exception as exc:
        st.error("Prediction failed because the model input did not match the saved training pipeline.")
        st.code(str(exc))
        st.caption("If this appears after deployment, check the scikit-learn/XGBoost versions and the saved feature schema.")
