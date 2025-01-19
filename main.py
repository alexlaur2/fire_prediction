import tkinter as tk
import webbrowser
from tkinter import Label, Button

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tkcalendar import DateEntry
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from itertools import product
import os

def generate_map():
    status_label.config(text="Se generează scenarii... Așteptați.", fg="blue")
    root.update_idletasks()

    # 1. Citire data selectată din DateEntry
    input_date_str = date_entry.get()
    try:
        input_date = pd.to_datetime(input_date_str)
    except ValueError:
        status_label.config(text="Data invalidă!", fg="red")
        return

    # 2. Încărcare dataset
    df = pd.read_csv("dataset.csv")
    print("Dataset Summary:")
    print(df.info(verbose=True))

    df = df.drop(columns=[
        "fire_year", "fire_number", "fire_name", "size_class", "fire_origin",
        "industry_identifier_desc", "responsible_group_desc", "activity_class", "true_cause",
        "det_agent_type", "det_agent", "dispatched_resource", "dispatch_date", "start_for_fire_date",
        "assessment_resource", "assessment_datetime", "assessment_hectares", "fire_spread_rate",
        "fire_type", "initial_action_by", "ia_arrival_at_fire_date", "ia_access", "fire_fighting_start_date",
        "fire_fighting_start_size", "bucketing_on_fire", "distance_from_water_source",
        "first_bucket_drop_date", "bh_fs_date", "bh_hectares", "uc_fs_date", "uc_hectares",
        "to_fs_date", "to_hectares", "ex_fs_date", "ex_hectares"
    ], errors="ignore")

    df.dropna(subset=["fire_location_latitude","fire_location_longitude"], inplace=True)
    df["reported_date"] = pd.to_datetime(df["reported_date"], errors="coerce")
    df["fire_start_date"] = pd.to_datetime(df["fire_start_date"], errors="coerce")
    df["discovered_date"] = pd.to_datetime(df["discovered_date"], errors="coerce")

    for col in ["temperature","relative_humidity","wind_speed","discovered_size"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".")
                .astype(float, errors="ignore")
            )

    df["weather_conditions_over_fire"] = df["weather_conditions_over_fire"].fillna("NA")
    df["wind_direction"] = df["wind_direction"].fillna("NA")
    df["general_cause_desc"] = df["general_cause_desc"].fillna("NA")
    df["fuel_type"] = df["fuel_type"].fillna("NA")

    dataset_summary = df.describe()
    print("Dataset Summary after transforming:")
    print(dataset_summary)
    print(df.info(verbose=True))

    # Visualization
    df.groupby(df["reported_date"].dt.date)[
        ["temperature", "relative_humidity", "wind_speed"]
    ].mean().plot(kind="bar", figsize=(12, 6))
    # plt.title("Average Temperature, Humidity, and Wind Speed Over Time")
    # plt.xlabel("Reported Date")
    # plt.ylabel("Values")
    # plt.legend(["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)"])
    # plt.tight_layout()
    # plt.savefig("bar_chart_avg_weather.png")
    # plt.show()
    #
    # fire_cause_distribution = df["general_cause_desc"].value_counts()
    # fire_cause_distribution.plot(kind="pie", autopct="%1.1f%%", figsize=(8, 8), startangle=90)
    # plt.title("Distribution of Fire Causes")
    # plt.ylabel("")
    # plt.tight_layout()
    # plt.savefig("pie_chart_fire_causes.png")
    # plt.show()

    # 3. Pregătire date pentru SARIMA
    num_cols_sarima = ["temperature","relative_humidity","wind_speed","discovered_size"]
    df_num = (
        df.groupby(df["reported_date"].dt.date)
          .agg({c:"mean" for c in num_cols_sarima})
          .reset_index()
          .rename(columns={"reported_date":"date"})
    )
    df_num["date"] = pd.to_datetime(df_num["date"])
    df_num.set_index("date", inplace=True)
    df_num = df_num.asfreq("D").interpolate()
    df_num = df_num.resample("W").mean()

    # Funcție încarcă/salvează modele SARIMA
    def load_or_train_sarima(serie, order, seasonal_order, model_name):
        model_path = f"models/sarima_{model_name}.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model_fit = pickle.load(f)
        else:
            sar_model = sm.tsa.statespace.SARIMAX(
                serie,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = sar_model.fit(disp=False)
            os.makedirs("models", exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(model_fit, f)

        return model_fit

    def plot_sarimax(series, sarima_model, steps=52, alpha=0.05):
        forecast_res = sarima_model.get_forecast(steps=steps)
        forecast_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=alpha)

        plt.figure(figsize=(10, 5))
        # Observate
        plt.plot(series.index, series, label="Observed", color="blue")
        # Forecast
        plt.plot(forecast_mean.index, forecast_mean, label="Forecast", color="orange")
        # CI
        plt.fill_between(forecast_mean.index,
                         conf_int.iloc[:, 0],
                         conf_int.iloc[:, 1],
                         color="orange", alpha=0.2,
                         label="Confidence Interval")

        plt.title(f"SARIMAX Forecast for {series.name}")
        plt.xlabel("Time")
        plt.ylabel(series.name)
        plt.legend()
        plt.tight_layout()
        plt.show()

    sarima_models = {}
    for c in num_cols_sarima:
        sarima_models[c] = load_or_train_sarima(
            df_num[c].interpolate(),
            (1,1,1),  # p,d,q
            (0,1,1,52),  # P,D,Q,m
            model_name=c
        )
        # plot_sarimax(df_num[c].interpolate(), sarima_models[c], steps=204, alpha=0.05)

    # 4. Clasificare (weather_conditions, wind_direction, general_cause_desc, fuel_type)
    cat_cols = ["weather_conditions_over_fire", "wind_direction", "general_cause_desc", "fuel_type"]
    encoders = {}
    for c in cat_cols:
        encoders[c] = LabelEncoder()
        df[c] = encoders[c].fit_transform(df[c].astype(str))

    df_cat = (
        df.groupby(df["reported_date"].dt.date)
        .agg({**{col: "mean" for col in cat_cols},
              **{col: "mean" for col in num_cols_sarima}})
        .reset_index()
        .rename(columns={"reported_date": "date"})
    )
    df_cat["date"] = pd.to_datetime(df_cat["date"])
    df_cat.set_index("date", inplace=True)
    df_cat = df_cat.asfreq("D").interpolate().resample("W").mean()

    cat_models = {}
    numeric_features_for_classif = num_cols_sarima

    def load_or_train_rf(X_train, y_train, model_name):
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train)
        return clf

    def evaluate_random_forest(clf, X_test, y_test, label):
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"Random Forest for: {label}")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1 Score:  {f1:.3f}")

    for c in cat_cols:
        temp_df = df_cat.dropna(subset=numeric_features_for_classif + [c])
        X_all = temp_df[numeric_features_for_classif]
        y_all = temp_df[c].round().astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                            test_size=0.4,
                                                            random_state=82)
        cat_models[c] = load_or_train_rf(X_train, y_train, f"clf_{c}")
        evaluate_random_forest(cat_models[c], X_test, y_test, label=c)

    # 5. Regresii (fire_start_date, discovered_date) + localizare (lat, lon)
    df_dates = df.dropna(subset=["fire_start_date","discovered_date","reported_date"]).copy()
    df_dates["offset_start"] = (df_dates["fire_start_date"] - df_dates["reported_date"]).dt.days
    df_dates["offset_disc"] = (df_dates["discovered_date"] - df_dates["reported_date"]).dt.days
    df_dates["lat"] = df_dates["fire_location_latitude"]
    df_dates["lon"] = df_dates["fire_location_longitude"]
    df_dates.dropna(subset=["offset_start","offset_disc","lat","lon"], inplace=True)

    df_dates_num = (
        df_dates.groupby(df_dates["reported_date"].dt.date)
                .agg({
                    "temperature":"mean",
                    "relative_humidity":"mean",
                    "wind_speed":"mean",
                    "discovered_size":"mean",
                    "offset_start":"mean",
                    "offset_disc":"mean",
                    "lat":"mean",
                    "lon":"mean"
                })
                .reset_index()
                .rename(columns={"reported_date":"date"})
    )
    df_dates_num["date"] = pd.to_datetime(df_dates_num["date"])
    df_dates_num.set_index("date", inplace=True)
    df_dates_num = df_dates_num.asfreq("D").interpolate().resample("W").mean()

    feat_reg = ["temperature", "relative_humidity", "wind_speed", "discovered_size"]

    def load_or_train_linreg(X_train, y_train, model_name="reg_model.pkl"):
        model_path = f"models/{model_name}"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                reg = pickle.load(f)
        else:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            os.makedirs("models", exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(reg, f)
        return reg

    def evaluate_regression(linreg, X_test, y_test, label=""):
        y_pred = linreg.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Regression Evaluation for {label}")
        print(f"MAE: {mae:.3f}")
        print(f"R^2: {r2:.3f}")

    targets = ["offset_start", "offset_disc", "lat", "lon"]
    reg_model = {}
    for target in targets:
        valid_df = df_dates_num.dropna(subset=feat_reg + [target])
        X_all = valid_df[feat_reg]
        y_all = valid_df[target]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=0.2,
            random_state=42
        )

        model_name = f"linreg_{target}.pkl"
        reg_model[target] = load_or_train_linreg(X_train, y_train, model_name)
        evaluate_regression(reg_model[target], X_test, y_test, label=target)

    # 6. Calcul steps
    last_known_date = df_num.index[-1]
    diff_days = (input_date - last_known_date).days
    steps_ahead = int(np.ceil(diff_days / 7.0))
    if steps_ahead < 0:
        status_label.config(text="Data în trecut, se folosește ultima predicție disponibilă!", fg="orange")
        steps_ahead = 0

    # generăm scenarii numeric (lower/upper)
    def get_lower_mean_upper(model_fit, steps):
        fc = model_fit.get_forecast(steps=steps)
        pm = fc.predicted_mean
        ci = fc.conf_int(alpha=0.2)
        if steps == 0:
            val_mean = model_fit.data.endog.iloc[-1]
            return [(val_mean, val_mean)]
        else:
            val_lower = ci.iloc[-1,0]
            val_upper = ci.iloc[-1,1]
            return [(val_lower, val_upper)]

    all_options = {}
    for c in num_cols_sarima:
        pair = get_lower_mean_upper(sarima_models[c], steps_ahead)
        val_lower, val_upper = pair[0]
        all_options[c] = [val_lower, val_upper]

    keys_cols = list(num_cols_sarima)
    combo_numeric = []
    def build_dict_numeric(vals):
        return {keys_cols[i]: vals[i] for i in range(len(keys_cols))}
    all_lists = [all_options[k] for k in keys_cols]
    for comb in product(*all_lists):
        combo_numeric.append(build_dict_numeric(comb))

    # 7. top_k categorii => combos
    top_k = 2
    cat_features_df_cols = numeric_features_for_classif
    cat_big_list = []

    for scenario_num in combo_numeric:
        row_df = pd.DataFrame([scenario_num])
        cats_probs_map = {}
        for c in cat_cols:
            model_c = cat_models[c]
            proba_c = model_c.predict_proba(row_df[cat_features_df_cols])[0]
            classes_c = model_c.classes_
            sorted_idx = np.argsort(proba_c)[::-1]
            top_idx = sorted_idx[:top_k]
            cat_list = [(classes_c[i], proba_c[i]) for i in top_idx]
            cats_probs_map[c] = cat_list
        cat_big_list.append((scenario_num, cats_probs_map))

    def inverse_label(categ, val):
        return encoders[categ].inverse_transform([int(val)])[0]

    combinations_all = []
    for (num_sc, cats_map) in cat_big_list:
        cat_keys = list(cat_cols)
        all_lists_cat = [cats_map[k] for k in cat_keys]
        for combo in product(*all_lists_cat):
            cat_values = [x[0] for x in combo]
            cat_probs = [x[1] for x in combo]
            prob_prod = np.prod(cat_probs)
            row = dict(num_sc)
            for i, ck in enumerate(cat_keys):
                row[ck] = inverse_label(ck, cat_values[i])
            row["prob_total"] = prob_prod
            combinations_all.append(row)

    combinations_all.sort(key=lambda x: x["prob_total"])

    # 8. offset + lat/lon + date
    result_rows = []
    for combo_row in combinations_all:
        x_lin = np.array([
            combo_row["temperature"],
            combo_row["relative_humidity"],
            combo_row["wind_speed"],
            combo_row["discovered_size"]
        ]).reshape(1,-1)
        offset_s = reg_model['offset_start'].predict(x_lin)[0]
        offset_d = reg_model['offset_disc'].predict(x_lin)[0]
        lat_pred = reg_model['lat'].predict(x_lin)[0]
        lon_pred = reg_model['lon'].predict(x_lin)[0]
        fire_start_pred = input_date + pd.Timedelta(days=offset_s)
        fire_disc_pred = input_date + pd.Timedelta(days=offset_d)
        row_out = {
            **combo_row,
            "fire_start_date": fire_start_pred,
            "discovered_date": fire_disc_pred,
            "fire_location_latitude": lat_pred,
            "fire_location_longitude": lon_pred
        }
        result_rows.append(row_out)

    df_scenarios = pd.DataFrame(result_rows)
    df_scenarios["prob_percent"] = df_scenarios["prob_total"]*100.0
    df_scenarios.to_csv("lista_scenarii.csv", index=False)

    m = folium.Map(location=[df_scenarios["fire_location_latitude"].mean(),
                             df_scenarios["fire_location_longitude"].mean()],
                   zoom_start=5)
    for i, row in df_scenarios.iterrows():
        popup_text = (
            f"Data început incendiu: {row['fire_start_date'].date()}<br>"
            f"Data actuală: {row['discovered_date'].date()}<br>"
            f"Condiții meteo: {row['weather_conditions_over_fire']}<br>"
            f"Direcție vânt: {row['wind_direction']}<br>"
            f"Cauză: {row['general_cause_desc']}<br>"
            f"Combustibil: {row['fuel_type']}<br>"
            f"Coordonate: ({row['fire_location_latitude']:.3f}, {row['fire_location_longitude']:.3f})"
        )
        folium.Marker(
            [row["fire_location_latitude"], row["fire_location_longitude"]],
            popup=popup_text,
            icon=folium.Icon(color="red", icon="fire"),
        ).add_to(m)

    m.save("lista_scenarii.html")

    msg = f"Scenarii generate"
    status_label.config(text=msg, fg="green")

    webbrowser.open("lista_scenarii.html")
    status_label.config(text="Harta a fost generată și deschisă în browser.")


# ------------------- UI Tkinter -------------------
import tkinter
from tkinter import Tk, Label, Button
from tkcalendar import DateEntry

root = Tk()
root.title("Aplicație Previziuni Incendii")

Label(root, text="Introduceți o dată (YYYY-MM-DD):").pack(pady=5)
date_entry = DateEntry(root, width=20, background='darkblue', foreground='white',
                       borderwidth=2, date_pattern='yyyy-mm-dd')
date_entry.pack(pady=5)

gen_button = Button(root, text="Generează Harta", command=generate_map)
gen_button.pack(pady=10)

status_label = Label(root, text="", fg="blue")
status_label.pack(pady=5)

root.mainloop()
