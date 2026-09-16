# Airline Passenger Satisfaction - Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Airline Passenger Satisfaction", page_icon="✈️", layout="wide")

# ---------------------------------------------------------------- palette
TEAL = "#2DD4BF"          # satisfied / positive
ROSE = "#FB7185"          # neutral or dissatisfied / negative
INDIGO = "#818CF8"        # secondary accent
SAT_COLORS = {"neutral or dissatisfied": ROSE, "satisfied": TEAL}
SEQ = ["#1E3A5F", "#2A7B9B", "#2DD4BF"]          # low -> high
DIVERGING = [ROSE, "#4B5563", TEAL]              # negative -> positive

SERVICE_COLS = ['Inflight wifi service', 'Ease of Online booking', 'Food and drink',
                'Online boarding', 'Seat comfort', 'Inflight entertainment',
                'On-board service', 'Leg room service', 'Baggage handling',
                'Checkin service', 'Inflight service', 'Cleanliness']

RATING_COLS = ['Inflight wifi service', 'Departure/Arrival time convenient',
               'Ease of Online booking', 'Gate location', 'Food and drink',
               'Online boarding', 'Seat comfort', 'Inflight entertainment',
               'On-board service', 'Leg room service', 'Baggage handling',
               'Checkin service', 'Inflight service', 'Cleanliness']


def style_fig(fig, height=330, title=None, showlegend=True):
    """One look for every chart: transparent background, tidy margins, compact height."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=showlegend,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        font=dict(size=12),
        hoverlabel=dict(font_size=12),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14)))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def f1_gradient(col):
    """Colour a column across the app palette without pulling in matplotlib."""
    stops = [(30, 58, 95), (42, 123, 155), (45, 212, 191)]   # SEQ as RGB
    lo, hi = float(col.min()), float(col.max())
    styles = []
    for v in col:
        t = 0.0 if hi == lo else (float(v) - lo) / (hi - lo)
        seg = 0 if t < 0.5 else 1
        f = t * 2 if seg == 0 else (t - 0.5) * 2
        a, b = stops[seg], stops[seg + 1]
        r, g, bl = (int(a[i] + (b[i] - a[i]) * f) for i in range(3))
        styles.append(f"background-color: rgb({r},{g},{bl}); color: white")
    return styles


@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')


@st.cache_data
def load_data():
    df = pd.read_csv('data/train.csv')
    df = df.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
    df['satisfied_num'] = (df['satisfaction'] == 'satisfied').astype(int)
    return df


@st.cache_data
def test_predictions():
    """Score the untouched test set so the home page can show a real confusion matrix."""
    raw = pd.read_csv('data/test.csv')
    y_true = (raw['satisfaction'] == 'satisfied').astype(int)
    x = raw.copy()
    x['Customer Type'] = x['Customer Type'].map({'Loyal Customer': 0, 'disloyal Customer': 1})
    x['Type of Travel'] = x['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
    x = x[list(transformer.feature_names_in_)]
    x = x.fillna(x.median(numeric_only=True))
    return y_true.values, model.predict(transformer.transform(x))


@st.cache_resource
def load_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)


saved = load_model()
model = saved['model']
transformer = saved['transformer']
model_name = saved['model_name']

st.sidebar.title("✈️ Airline Satisfaction")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🔮 Prediction"])
st.sidebar.markdown("---")
st.sidebar.caption("Predicting airline passenger satisfaction from trip details "
                   "and service ratings.")


# -------------------------------------------------------------------- Home
if page == "🏠 Home":
    st.title("✈️ Airline Passenger Satisfaction")
    st.write("Predicting whether an airline passenger will be satisfied, from their trip "
             "details and the ratings they give to each service.")

    try:
        comparison = pd.read_csv('model_comparison.csv', index_col=0)
        best = comparison.loc[model_name]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model", model_name.replace("Tuned ", ""))
        c2.metric("Test Accuracy", f"{best['Test Accuracy']*100:.2f} %")
        c3.metric("F1 Score", f"{best['F1 Score']*100:.2f} %")
        c4.metric("Overfitting Gap", f"{(best['Train Accuracy']-best['Test Accuracy'])*100:.2f} %")
        st.caption(f"Chosen by F1 score out of {len(comparison)} trained models.")
    except FileNotFoundError:
        comparison = None
        st.info("Run the notebook cell that saves model_comparison.csv to see the comparison.")

    st.markdown("---")
    st.header("Key Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🛫 Online boarding leads")
        st.write("It shows the widest rating gap between satisfied and dissatisfied "
                 "passengers, and the model leans on it more than any other feature.")
    with c2:
        st.markdown("#### 💼 Business travel wins")
        st.write("Business travellers and loyal customers are far more satisfied than "
                 "personal travellers and first-time flyers.")
    with c3:
        st.markdown("#### ⏱️ Delays matter less")
        st.write("Median delay is 0 minutes for both groups, so delay is a weak signal "
                 "next to the service ratings.")

    st.markdown("---")
    st.header("Model Performance")

    if comparison is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("All models")
            st.dataframe(comparison.style.format("{:.4f}")
                         .apply(f1_gradient, subset=['F1 Score']),
                         width='stretch', height=330)
        with c2:
            st.subheader("F1 score by model")
            ordered = comparison.sort_values('F1 Score')
            data = pd.DataFrame({'Model': ordered.index, 'F1 Score': ordered['F1 Score'].values})
            fig = px.bar(data, x='F1 Score', y='Model', orientation='h',
                         color='F1 Score', color_continuous_scale=SEQ)
            fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
            fig.update_xaxes(range=[0, 1])
            st.plotly_chart(style_fig(fig, 330, showlegend=False), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Where it gets things wrong")
        y_true, y_pred = test_predictions()
        cm = pd.crosstab(pd.Series(y_true, name='Actual'), pd.Series(y_pred, name='Predicted'))
        labels = ['Dissatisfied', 'Satisfied']
        fig = px.imshow(cm.values, text_auto='d', color_continuous_scale=SEQ,
                        x=labels, y=labels, labels=dict(x="Predicted", y="Actual"))
        fig.update_traces(textfont_size=18, hovertemplate=
                          "Actual %{y}, predicted %{x}: %{z:,}<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickangle=0, side='bottom')
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(style_fig(fig, 340, showlegend=False), use_container_width=True)
        errors = int((y_true != y_pred).sum())
        st.caption(f"On the {len(y_true):,} unseen test passengers the model gets "
                   f"**{errors:,}** wrong ({errors/len(y_true)*100:.2f}%).")
    with c2:
        if hasattr(model, 'feature_importances_'):
            st.subheader("What the model relies on")
            names = [c.split('__')[-1] for c in transformer.get_feature_names_out()]
            imp = pd.Series(model.feature_importances_, index=names).sort_values().tail(10)
            data = pd.DataFrame({'Feature': imp.index, 'Importance': imp.values})
            fig = px.bar(data, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale=SEQ)
            fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
            st.plotly_chart(style_fig(fig, 340, showlegend=False), use_container_width=True)
            st.caption(f"The strongest driver of satisfaction is **{imp.index[-1]}**.")

    st.markdown("---")
    st.header("How It Works")
    c1, c2, c3, c4 = st.columns(4)
    for col, (step, text) in zip([c1, c2, c3, c4], [
            ("1 · Clean", "Drop the id column, fill missing arrival delays with the median, "
                          "and check for duplicates."),
            ("2 · Explore", "Seven questions about who is satisfied and which services "
                            "separate the two groups."),
            ("3 · Prepare", "Drop five weakly correlated features, scale the numbers and "
                            "one-hot encode the class."),
            ("4 · Model", "Train and tune several models, then keep the one with the "
                          "best F1 score.")]):
        with col:
            st.markdown(f"**{step}**")
            st.caption(text)

    st.markdown("---")
    st.caption("Use the sidebar to explore the data or predict a passenger's satisfaction.")


# --------------------------------------------------------------------- EDA
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    df = load_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Passengers", f"{len(df):,}")
    c2.metric("Satisfied", f"{df['satisfied_num'].mean()*100:.1f} %")
    c3.metric("Features", df.shape[1] - 2)
    c4.metric("Average age", f"{df['Age'].mean():.0f}")
    st.markdown("---")

    # Q1
    st.subheader("Q1: Is the target balanced?")
    c1, c2 = st.columns(2)
    counts = df['satisfaction'].value_counts()
    with c1:
        fig = px.pie(values=counts.values, names=counts.index, hole=0,
                     color=counts.index, color_discrete_map=SAT_COLORS)
        fig.update_traces(textposition='inside', textinfo='percent')
        st.plotly_chart(style_fig(fig, 300, "Share"), use_container_width=True)
    with c2:
        data = pd.DataFrame({'satisfaction': counts.index, 'Passengers': counts.values})
        fig = px.bar(data, x='satisfaction', y='Passengers', color='satisfaction',
                     color_discrete_map=SAT_COLORS)
        fig.update_layout(xaxis_title=None)
        st.plotly_chart(style_fig(fig, 300, "Count", showlegend=False), use_container_width=True)
    st.caption(f"The target leans slightly to dissatisfied: "
               f"**{(1-df['satisfied_num'].mean())*100:.1f}%** against "
               f"**{df['satisfied_num'].mean()*100:.1f}%** satisfied — close enough to balanced "
               "that accuracy and F1 stay meaningful.")

    # Q2
    st.subheader("Q2: Which passenger groups are most satisfied?")
    overall = df['satisfied_num'].mean() * 100
    groups = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    for pair in [groups[:2], groups[2:]]:
        for col_box, group in zip(st.columns(2), pair):
            rate = (df.groupby(group)['satisfied_num'].mean().sort_values() * 100)
            data = pd.DataFrame({group: rate.index.astype(str), '% satisfied': rate.values})
            fig = px.bar(data, x='% satisfied', y=group, orientation='h',
                         color='% satisfied', color_continuous_scale=SEQ)
            fig.add_vline(x=overall, line_dash="dash", line_color="grey")
            fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
            fig.update_xaxes(range=[0, 100])
            with col_box:
                st.plotly_chart(style_fig(fig, 250, group, showlegend=False),
                                use_container_width=True)
    st.caption(f"Business travellers and loyal customers stand out; the dashed line is the "
               f"overall rate of {overall:.1f}%. Gender barely moves the needle.")

    # Q3
    st.subheader("Q3: Does age affect satisfaction?")
    age_bins = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100])
    age_sat = df.groupby(age_bins, observed=True)['satisfied_num'].mean() * 100
    data = pd.DataFrame({'Age group': age_sat.index.astype(str), '% satisfied': age_sat.values})
    fig = px.bar(data, x='Age group', y='% satisfied',
                 color='% satisfied', color_continuous_scale=SEQ)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 300, showlegend=False), use_container_width=True)
    st.caption("Satisfaction peaks in the 45-60 group and drops sharply for the youngest "
               "and oldest passengers.")

    # Q4
    st.subheader("Q4: Which services separate satisfied from dissatisfied passengers?")
    gap = []
    for col in RATING_COLS:
        s = df.loc[(df['satisfied_num'] == 1) & (df[col] > 0), col].mean()
        d = df.loc[(df['satisfied_num'] == 0) & (df[col] > 0), col].mean()
        gap.append({'service': col, 'satisfied': s, 'dissatisfied': d, 'gap': s - d})
    gap_df = pd.DataFrame(gap).sort_values('gap')
    fig = px.bar(gap_df, x='gap', y='service', orientation='h',
                 color='gap', color_continuous_scale=DIVERGING, color_continuous_midpoint=0,
                 labels={'gap': 'Difference in average rating', 'service': ''})
    fig.add_vline(x=0, line_color="grey")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 380, showlegend=False), use_container_width=True)
    top = gap_df.iloc[-1]
    st.caption(f"**{top['service']}** shows the widest gap ({top['gap']:.2f} rating points), "
               "while gate location and time convenience separate the groups the least.")

    # Q5
    st.subheader("Q5: Why does a 0 rating look so strange?")
    cols = st.columns(3)
    for col_box, col in zip(cols, ['Inflight wifi service', 'Ease of Online booking', 'Online boarding']):
        rate = df.groupby(col)['satisfied_num'].mean() * 100
        data = pd.DataFrame({'Rating': rate.index.astype(str), '% satisfied': rate.values})
        fig = px.bar(data, x='Rating', y='% satisfied',
                     color='% satisfied', color_continuous_scale=SEQ)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_yaxes(range=[0, 105])
        with col_box:
            st.plotly_chart(style_fig(fig, 260, col, showlegend=False), use_container_width=True)
    zero_n = int((df['Inflight wifi service'] == 0).sum())
    zero_rate = df.loc[df['Inflight wifi service'] == 0, 'satisfied_num'].mean() * 100
    st.caption(f"A 0 does not behave like a low score: the **{zero_n:,}** passengers who rated "
               f"wifi 0 are **{zero_rate:.1f}%** satisfied, far above those who rated it 1. "
               "A 0 means the service was not used rather than badly rated.")

    # Q6
    st.subheader("Q6: Do flight delays make passengers unhappy?")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x='satisfaction', y='Departure Delay in Minutes',
                     color='satisfaction', color_discrete_map=SAT_COLORS)
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(xaxis_title=None)
        st.plotly_chart(style_fig(fig, 300, "Departure delay (0-100 min)", showlegend=False),
                        use_container_width=True)
    with c2:
        delay = df.groupby('satisfaction')[['Departure Delay in Minutes',
                                            'Arrival Delay in Minutes']].mean().reset_index()
        delay = delay.melt(id_vars='satisfaction', var_name='Delay', value_name='Minutes')
        fig = px.bar(delay, x='Delay', y='Minutes', color='satisfaction',
                     barmode='group', color_discrete_map=SAT_COLORS)
        fig.update_layout(xaxis_title=None)
        st.plotly_chart(style_fig(fig, 300, "Average delay"), use_container_width=True)
    st.caption("Median delay is 0 minutes for both groups and the means differ by only a few "
               "minutes, so delay is a weak signal compared with the service ratings.")

    # Q7
    st.subheader("Q7: Do age and flight distance differ between the two groups?")
    c1, c2 = st.columns(2)
    for col_box, col in zip([c1, c2], ['Age', 'Flight Distance']):
        fig = px.box(df, x='satisfaction', y=col, color='satisfaction',
                     color_discrete_map=SAT_COLORS)
        fig.update_layout(xaxis_title=None)
        with col_box:
            st.plotly_chart(style_fig(fig, 300, col, showlegend=False), use_container_width=True)
    st.caption("Satisfied passengers skew a little older and fly noticeably longer distances, "
               "which fits the business-travel pattern seen in Q2.")


# -------------------------------------------------------------- Prediction
else:
    st.title("🔮 Passenger Satisfaction Prediction")
    tab_one, tab_batch = st.tabs(["One passenger", "Upload a CSV"])

    with tab_one:
     st.write("Fill in the passenger details below to predict whether they will be satisfied.")

     with st.form("passenger_form"):
         st.subheader("Passenger Information")
         c1, c2, c3 = st.columns(3)
         with c1:
             age = st.number_input("Age", min_value=1, max_value=100, value=40)
             customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
         with c2:
             type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
             travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
         with c3:
             flight_distance = st.number_input("Flight Distance", min_value=1, max_value=5000, value=1200)

         st.subheader("Service Ratings")
         st.caption("0 means not applicable, 1 is the worst and 5 is the best.")
         ratings = {}
         cols = st.columns(3)
         for i, col_name in enumerate(SERVICE_COLS):
             with cols[i % 3]:
                 ratings[col_name] = st.slider(col_name, 0, 5, 3)

         submitted = st.form_submit_button("Predict Satisfaction", width='stretch')

     if submitted:
         # Build one row in exactly the format the transformer was fitted on
         row = {'Customer Type': 0 if customer_type == "Loyal Customer" else 1,
                'Age': age,
                'Type of Travel': 1 if type_of_travel == "Business travel" else 0,
                'Class': travel_class,
                'Flight Distance': flight_distance}
         row.update(ratings)
         new_data = pd.DataFrame([row])[list(transformer.feature_names_in_)]

         prepared = transformer.transform(new_data)
         prediction = int(model.predict(prepared)[0])
         proba = model.predict_proba(prepared)[0]

         st.markdown("---")
         if prediction == 1:
             st.success("### Prediction : Satisfied 😀")
         else:
             st.error("### Prediction : Neutral or Dissatisfied 😕")

         c1, c2 = st.columns(2)
         c1.metric("Probability of being satisfied", f"{proba[1]*100:.2f} %")
         c2.metric("Probability of being dissatisfied", f"{proba[0]*100:.2f} %")
         st.progress(float(proba[1]))

         # ------------------------------------------- why the model said that
         st.subheader("What drove this prediction")
         try:
             explainer = load_explainer(model)
             shap_values = np.array(explainer.shap_values(prepared))
             if shap_values.ndim == 3:            # some models return one set per class
                 shap_values = shap_values[:, :, 1]
             contrib = pd.Series(shap_values[0],
                                 index=[c.split('__')[-1] for c in transformer.get_feature_names_out()])
             contrib = contrib.reindex(contrib.abs().sort_values().index).tail(8)

             fig = go.Figure(go.Bar(
                 x=contrib.values, y=contrib.index, orientation='h',
                 marker_color=[TEAL if v > 0 else ROSE for v in contrib.values],
                 hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
             fig.add_vline(x=0, line_color="grey")
             fig.update_layout(xaxis_title="← dissatisfied      contribution      satisfied →")
             st.plotly_chart(style_fig(fig, 340, showlegend=False), use_container_width=True)
             st.caption("Exact TreeSHAP contributions for *this* passenger — "
                        "teal pushes toward satisfied, rose toward dissatisfied.")
         except Exception:
             st.info("Install shap to see the per-passenger explanation (pip install shap).")

    with tab_batch:
        st.write("Upload a CSV of passengers to score them all at once. "
                 "It needs the same columns as the original dataset — "
                 "`data/test.csv` works as is.")

        uploaded = st.file_uploader("Choose a CSV file", type="csv")

        with st.expander("Which columns are required?"):
            st.write(", ".join(f"`{c}`" for c in transformer.feature_names_in_))
            st.caption("`Customer Type` and `Type of Travel` may be text "
                       "(Loyal Customer / Business travel) or already encoded as 0/1.")

        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            missing = [c for c in transformer.feature_names_in_ if c not in raw.columns]
            if missing:
                st.error("The file is missing these columns: " + ", ".join(missing))
            else:
                x = raw.copy()
                # accept either the raw text labels or the already encoded 0/1
                if x['Customer Type'].dtype == object:
                    x['Customer Type'] = x['Customer Type'].map(
                        {'Loyal Customer': 0, 'disloyal Customer': 1})
                if x['Type of Travel'].dtype == object:
                    x['Type of Travel'] = x['Type of Travel'].map(
                        {'Personal Travel': 0, 'Business travel': 1})
                x = x[list(transformer.feature_names_in_)]
                x = x.fillna(x.median(numeric_only=True))

                prepared = transformer.transform(x)
                preds = model.predict(prepared)
                probs = model.predict_proba(prepared)[:, 1]

                out = raw.copy()
                out['Prediction'] = pd.Series(preds).map(
                    {0: 'neutral or dissatisfied', 1: 'satisfied'}).values
                out['Probability satisfied'] = probs.round(4)

                st.success(f"Scored {len(out):,} passengers.")
                c1, c2, c3 = st.columns(3)
                c1.metric("Passengers", f"{len(out):,}")
                c2.metric("Predicted satisfied", f"{(preds == 1).sum():,}")
                c3.metric("Satisfaction rate", f"{preds.mean()*100:.1f} %")

                c1, c2 = st.columns([1, 1])
                with c1:
                    share = out['Prediction'].value_counts()
                    fig = px.pie(values=share.values, names=share.index, hole=0.45,
                                 color=share.index, color_discrete_map=SAT_COLORS)
                    fig.update_traces(textposition='inside', textinfo='percent')
                    st.plotly_chart(style_fig(fig, 300, "Predicted split"),
                                    use_container_width=True)
                with c2:
                    fig = px.histogram(out, x='Probability satisfied', nbins=40,
                                       color_discrete_sequence=[TEAL])
                    fig.update_layout(yaxis_title="Passengers")
                    st.plotly_chart(style_fig(fig, 300, "How confident the model is",
                                              showlegend=False), use_container_width=True)

                # if the file already has the answer, show how well the model did
                if 'satisfaction' in raw.columns:
                    truth = raw['satisfaction']
                    truth = (truth == 'satisfied').astype(int) if truth.dtype == object else truth
                    st.info(f"The file included the real answer — accuracy on these "
                            f"{len(out):,} rows is **{(truth.values == preds).mean()*100:.2f} %**.")

                st.dataframe(out.head(100), width='stretch', height=300)
                st.caption("Showing the first 100 rows. The download has every row.")
                st.download_button("Download all predictions",
                                   out.to_csv(index=False).encode('utf-8'),
                                   file_name="predictions.csv", mime="text/csv",
                                   width='stretch')

    st.markdown("---")
    st.caption(f"Predictions come from the {model_name} model "
               "trained in the accompanying notebook.")
