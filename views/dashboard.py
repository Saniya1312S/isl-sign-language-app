"""Page 3 - Dashboard: dataset analytics + honest model evaluation. Reads results/*.json only."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from isl import ui
from isl.config import ALPHABET, RESULTS_DIR
from isl.runtime import read_result

TEAL, CYAN, AMBER, ROSE, VIOLET, GREY = "#14B8A6", "#22D3EE", "#F59E0B", "#FB7185", "#A78BFA", "#64748B"
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def style(fig, height=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13))
    return fig


def need(data, script):
    if data is None:
        st.info(f"Not generated yet. Run `{script}` (see README).")
        return False
    return True


def pct(x):
    return f"{x:.1%}"


# ---------------------------------------------------------------------------------------- tabs
def tab_dataset(eda):
    if not need(eda, "python scripts/eda.py"):
        return
    src = eda["sources"]
    cols = st.columns(4)
    ui.kpi(cols[0], f"{eda['total_images']:,}", "images processed", "all sources")
    ui.kpi(cols[1], "26", "classes (A-Z)", "Indian Sign Language alphabet")
    for c, (name, v) in zip(cols[2:], src.items()):
        ui.kpi(c, pct(v["detection_rate"]), f"hand found · {name}", f"{v['images']:,} images · {v['image_size']} px")

    st.markdown("#### Images per class")
    rows = [{"class": k, "source": s, "images": n} for s, v in src.items() for k, n in v["per_class"].items()]
    fig = px.bar(pd.DataFrame(rows), x="class", y="images", color="source", barmode="stack",
                 color_discrete_sequence=[TEAL, AMBER], category_orders={"class": ALPHABET})
    st.plotly_chart(style(fig, 330), width="stretch")
    ui.note("The main (Indian) source is almost perfectly balanced at ~1,200 images per letter; the external source has only "
            "21–46 per letter and no H, J or Y. Always check balance before choosing accuracy as a metric.", "good")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### One hand or two?")
        main_src = "Indian"
        hb = src[main_src]["hands_by_class"]
        rows = [{"class": k, "hands": {"0": "none found", "1": "1 hand", "2": "2 hands"}[h], "images": n}
                for k, d in hb.items() for h, n in d.items()]
        fig = px.bar(pd.DataFrame(rows), x="class", y="images", color="hands", barmode="stack",
                     color_discrete_map={"1 hand": CYAN, "2 hands": TEAL, "none found": ROSE}, category_orders={"class": ALPHABET})
        st.plotly_chart(style(fig, 330), width="stretch")
        st.caption("Many ISL letters use both hands, so a model that only reads one hand cannot work for the full alphabet.")
    with c2:
        st.markdown("#### Is MediaPipe's Left/Right label a reliable feature?")
        h = eda["handedness"].get("Indian")
        if h:
            df = pd.DataFrame({"class": list(h["left_share_by_class"]), "share labelled Left": list(h["left_share_by_class"].values())})
            fig = px.bar(df, x="class", y="share labelled Left", color_discrete_sequence=[AMBER], category_orders={"class": ALPHABET})
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(style(fig, 330), width="stretch")
            st.caption(f"One-hand letters only. {h['classes_split_both_ways']} of {h['n_classes']} letters are labelled both "
                       "Left and Right for the *same* sign. The original project used this label to choose a feature slot, "
                       "so identical signs landed in different columns. The new pipeline ignores the label.")

    st.markdown("#### Near-duplicate check — why a random split is misleading")
    nd = eda["near_duplicates"]
    c = st.columns(4)
    ui.kpi(c[0], pct(nd["share_nn_is_within_2_frames"]), "nearest twin ≤ 2 frames away", "random-split view")
    ui.kpi(c[1], f"{nd['median_nn_distance_random_split']}", "median distance to nearest twin", "random split")
    ui.kpi(c[2], f"{nd['median_nn_distance_block_split']}", "…when whole blocks are held out", f"block = {nd['block_size_frames']} frames")
    ui.kpi(c[3], f"×{nd['ratio_block_over_random']}", "farther with block split", "less leakage")
    st.caption(nd["interpretation"])

    st.markdown("#### How the two data sources differ (PCA of the 171 features)")
    p = RESULTS_DIR / "pca_sample.csv"
    if p.exists():
        pca = pd.read_csv(p)
        colour = st.radio("Colour by", ["source", "label"], horizontal=True)
        fig = px.scatter(pca, x="pc1", y="pc2", color=colour, symbol="source", opacity=0.75,
                         color_discrete_sequence=px.colors.qualitative.Set2 if colour == "source" else px.colors.qualitative.Alphabet,
                         hover_data=["label", "n_hands"])
        ev = eda.get("pca_explained_variance", [0, 0])
        fig.update_layout(xaxis_title=f"PC1 ({ev[0]:.0%} of variance)", yaxis_title=f"PC2 ({ev[1]:.0%})")
        st.plotly_chart(style(fig, 460), width="stretch")

    st.markdown("#### What the model 'sees': average normalised hand shape per letter")
    shapes = eda["mean_shapes"]
    letter = st.select_slider("Letter", options=sorted(shapes), value="A")
    s = shapes[letter]
    fig = go.Figure()
    for hnd in range(s["n_hands"]):
        pts = np.array(s["points"][hnd])
        for a, b in BONES:
            fig.add_trace(go.Scatter(x=[pts[a, 0], pts[b, 0]], y=[-pts[a, 1], -pts[b, 1]], mode="lines",
                                     line=dict(color=TEAL if hnd == 0 else CYAN, width=3), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pts[:, 0], y=-pts[:, 1], mode="markers", marker=dict(size=7, color=AMBER),
                                 showlegend=False, hovertext=[f"landmark {i}" for i in range(21)]))
    fig.update_yaxes(scaleanchor="x", visible=False)
    fig.update_xaxes(visible=False)
    st.plotly_chart(style(fig, 360), width="stretch")
    st.caption(f"Letter {letter}: {s['n_hands']} hand(s). Wrist-centred and size-normalised, so position and distance from the "
               "camera no longer matter.")


def tab_protocols(protocols, ablation, learning, adaptation):
    st.markdown("### The same model, three ways of measuring it")
    if need(protocols, "python scripts/train_alphabet.py --stage protocols"):
        rows = [("P1 · random split\n(what the original did)", protocols["P1_random_split"]["accuracy"], ROSE),
                ("P2 · block split\n(unseen video segments)", protocols["P2_block_split"]["accuracy"], AMBER),
                ("P3 · external dataset\n(other people & backgrounds)", protocols["P3_external"]["accuracy"], TEAL)]
        fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=[r[1] for r in rows], marker_color=[r[2] for r in rows],
                               text=[pct(r[1]) for r in rows], textposition="outside"))
        fig.update_yaxes(range=[0, 1.08], tickformat=".0%", title="accuracy")
        st.plotly_chart(style(fig, 380), width="stretch")
        e = protocols["P3_external"]
        ui.note(f"Same model, same features — only the **test protocol** changes. A random split rewards memorising neighbouring "
                f"video frames. On the external set MediaPipe also finds a hand in only {e['n_images_with_hand']} of "
                f"{e['n_images_total']} images, so the **end-to-end** accuracy (missed hands count as wrong) is {pct(e['end_to_end_accuracy'])}.")
    st.markdown("### Ablation: which design choices matter?")
    if need(ablation, "python scripts/train_alphabet.py --stage ablation"):
        rows = [{"features": r["features"], "mirror augmentation": "yes" if r["mirror_augmentation"] else "no",
                 "# features": r["n_features"], "block-split acc.": r["block_test"]["accuracy"],
                 "external acc.": r["external"]["accuracy"]} for r in ablation["rows"]]
        df = pd.DataFrame(rows)
        melt = df.melt(id_vars=["features", "mirror augmentation", "# features"], value_vars=["block-split acc.", "external acc."],
                       var_name="test", value_name="accuracy")
        melt["setup"] = melt["features"] + " · mirror " + melt["mirror augmentation"]
        fig = px.bar(melt, x="setup", y="accuracy", color="test", barmode="group", color_discrete_sequence=[AMBER, TEAL])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 400), width="stretch")
        st.dataframe(df.style.format({"block-split acc.": "{:.1%}", "external acc.": "{:.1%}"}), hide_index=True, width="stretch")
        st.caption("legacy_raw126 = the original features (raw coordinates, slot chosen by MediaPipe's Left/Right label). "
                   "norm / norm_eng = wrist-centred, size-normalised (+ engineered finger features).")
    st.markdown("### How much data is enough? (learning curve)")
    if need(learning, "python scripts/train_alphabet.py --stage learning"):
        df = pd.DataFrame(learning["rows"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.images_per_class, y=df.block_test_accuracy, name="held-out blocks", mode="lines+markers", line=dict(color=AMBER)))
        fig.add_trace(go.Scatter(x=df.images_per_class, y=df.external_accuracy, name="external dataset", mode="lines+markers", line=dict(color=TEAL)))
        fig.update_xaxes(type="log", title="training images per class")
        fig.update_yaxes(tickformat=".0%", title="accuracy")
        st.plotly_chart(style(fig, 360), width="stretch")
        st.caption("More studio images barely help on the other dataset: the missing ingredient is **variety** (people, rooms, "
                   "cameras), not quantity.")
    st.markdown("### What a little data from the new person/room can do (personalisation)")
    if need(adaptation, "python scripts/adaptation_study.py"):
        df = pd.DataFrame(adaptation["rows"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.calibration_images_per_letter, y=df.external_accuracy, name="new setting (external, held-out)",
                                 mode="lines+markers", line=dict(color=TEAL, width=3)))
        fig.add_trace(go.Scatter(x=df.calibration_images_per_letter, y=df.studio_block_accuracy, name="original studio data",
                                 mode="lines+markers", line=dict(color=AMBER)))
        fig.update_xaxes(title="example images per letter added from the new setting", dtick=1)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.02], title="accuracy")
        st.plotly_chart(style(fig, 360), width="stretch")
        a0, a8 = df.iloc[0], df.iloc[-1]
        ui.note(f"With **{int(a8.calibration_images_per_letter)} example images per letter** the accuracy on the new setting rises from "
                f"{pct(a0.external_accuracy)} to {pct(a8.external_accuracy)}, while accuracy on the original data stays about "
                f"{pct(a8.studio_block_accuracy)}. This is the evidence behind the **“Teach the app your hands”** feature. "
                f"Protocol: {adaptation['protocol']}.", "good")


def tab_models(comp):
    if not need(comp, "python scripts/train_alphabet.py --stage compare"):
        return
    st.markdown(f"### Six classifiers, one fair comparison")
    st.caption(f"Features: {comp['features']} · selection by {comp['selected_by']} · test numbers touched only after selection.")
    df = pd.DataFrame([{
        "model": r["model"], "CV macro-F1": r["cv_macro_f1"], "train acc.": r["train_accuracy"],
        "block-test acc.": r["block_test"]["accuracy"], "external acc.": r["external"]["accuracy"],
        "gap (train − test)": r["generalisation_gap"], "fit (s)": r["fit_seconds"],
        "predict (ms/sample)": r["predict_ms_per_sample"], "size (MB)": r.get("model_size_mb")} for r in comp["rows"]])
    fig = px.bar(df.melt(id_vars="model", value_vars=["CV macro-F1", "block-test acc.", "external acc."]),
                 x="model", y="value", color="variable", barmode="group", color_discrete_sequence=[VIOLET, AMBER, TEAL])
    fig.update_yaxes(tickformat=".0%", title="")
    st.plotly_chart(style(fig, 400), width="stretch")
    fmt = {c: "{:.1%}" for c in ["CV macro-F1", "train acc.", "block-test acc.", "external acc.", "gap (train − test)"]}
    fmt.update({"fit (s)": "{:.1f}", "predict (ms/sample)": "{:.3f}", "size (MB)": "{:.2f}"})
    st.dataframe(df.style.format(fmt, na_rep="–").highlight_max(subset=["block-test acc.", "external acc."], color="#134e4a"),
                 hide_index=True, width="stretch")
    st.success(f"Selected model: **{comp['selected_model']}**")
    st.caption("A large train − test gap means the model memorised its training frames. The MLP row is the 'neural network' "
               "comparison: it is a small scikit-learn network, not a deep-learning model.")


def tab_errors(rep):
    if not need(rep, "python scripts/train_alphabet.py --stage final"):
        return
    which = st.radio("Evaluate on", ["block_test", "external"], horizontal=True,
                     format_func=lambda k: {"block_test": "Held-out blocks (Indian dataset)", "external": "External dataset (ISL_Dataset_2)"}[k])
    r = rep[which]
    labels, cm = r["labels"], np.array(r["confusion_matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig = px.imshow(norm, x=labels, y=labels, color_continuous_scale="Teal", zmin=0, zmax=1, aspect="auto",
                    labels=dict(x="predicted", y="true", color="share"))
    fig.update_traces(hovertemplate="true %{y} → predicted %{x}: %{z:.0%}<extra></extra>")
    st.plotly_chart(style(fig, 560), width="stretch")
    c1, c2 = st.columns([3, 2])
    with c1:
        pc = pd.DataFrame(r["per_class"]).sort_values("f1")
        fig = px.bar(pc, x="f1", y="label", orientation="h", color="f1", color_continuous_scale="Teal", range_color=[0, 1])
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(dtick=1, autorange="reversed"))
        st.markdown("**Per-letter F1 (weakest first)**")
        st.plotly_chart(style(fig, 640), width="stretch")
    with c2:
        st.markdown("**Most common confusions**")
        st.dataframe(pd.DataFrame(r["top_confusions"]).rename(columns={"true": "true letter", "pred": "mistaken for"}),
                     hide_index=True, width="stretch")
        if "importance_by_group" in rep:
            st.markdown("**Which features matter (model importance)**")
            g = pd.Series(rep["importance_by_group"]).sort_values()
            fig = px.bar(x=g.values, y=g.index, orientation="h", color_discrete_sequence=[CYAN])
            fig.update_layout(xaxis_title="share of importance", yaxis_title="")
            st.plotly_chart(style(fig, 260), width="stretch")

    st.markdown("#### Confidence threshold: accuracy vs. how often the app answers")
    curve = pd.DataFrame(r["confidence_curve"])
    t = st.slider("Minimum confidence", 0.0, 0.95, 0.60, 0.05)
    row = curve.iloc[(curve.threshold - t).abs().argmin()]
    c = st.columns(3)
    ui.kpi(c[0], pct(row.accuracy) if row.accuracy is not None else "–", "accuracy of answered frames")
    ui.kpi(c[1], pct(row.coverage), "frames that get an answer")
    ui.kpi(c[2], pct(1 - row.coverage), "frames the app stays silent on")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.threshold, y=curve.accuracy, name="accuracy", line=dict(color=TEAL)))
    fig.add_trace(go.Scatter(x=curve.threshold, y=curve.coverage, name="coverage", line=dict(color=AMBER)))
    fig.add_vline(x=t, line_dash="dot", line_color=GREY)
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title="minimum confidence")
    st.plotly_chart(style(fig, 320), width="stretch")
    st.caption("A higher threshold makes the app more accurate but more hesitant. In a live app the hold-time rule adds a second "
               "safety net, because a letter must win the vote on many consecutive frames before it is typed.")


def tab_performance(lat, legacy):
    st.markdown("### Speed")
    if need(lat, "python scripts/benchmark_latency.py"):
        stages = lat["stages_ms_p50"]
        fig = go.Figure(go.Bar(x=list(stages.values()), y=list(stages), orientation="h", marker_color=TEAL,
                               text=[f"{v:.1f} ms" for v in stages.values()], textposition="auto"))
        fig.update_layout(xaxis_title="milliseconds per frame (median)")
        st.plotly_chart(style(fig, 260), width="stretch")
        c = st.columns(3)
        ui.kpi(c[0], f"{lat['total_ms_p50']:.0f} ms", "median per frame", f"p95 {lat['total_ms_p95']:.0f} ms")
        ui.kpi(c[1], f"{1000 / lat['total_ms_p50']:.0f}", "frames per second (max)", "single CPU core")
        ui.kpi(c[2], f"{lat['classifier_ms_p50']:.2f} ms", "classifier alone", "after landmarks")
        st.caption(lat["note"])
    st.markdown("### Original pipeline vs. new pipeline")
    if need(legacy, "python scripts/evaluate_legacy.py"):
        rows = []
        for r in legacy["rows"]:
            rows.append({"model": r["model"], "test condition": r["scenario"], "dataset": r["dataset"],
                         "accuracy": r["accuracy"], "images": r["n"]})
        df = pd.DataFrame(rows)
        fig = px.bar(df, x="test condition", y="accuracy", color="model", barmode="group", facet_col="dataset",
                     color_discrete_sequence=[ROSE, TEAL])
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(tickangle=20, title="")
        st.plotly_chart(style(fig, 440), width="stretch")
        st.dataframe(df.style.format({"accuracy": "{:.1%}"}), hide_index=True, width="stretch")
        st.caption(legacy["note"])


def main() -> None:
    ui.inject_css()
    ui.hero("📊 Data & Model Dashboard",
            "Dataset analytics and an honest model evaluation. Every number here is read from files produced by the scripts in "
            "`scripts/` — nothing is typed in by hand.")
    eda = read_result("eda.json")
    t1, t2, t3, t4, t5 = st.tabs(["🗂️ Dataset", "🧪 Evaluation honesty", "🏁 Models", "🔍 Errors & confidence", "⚡ Speed & old-vs-new"])
    with t1:
        tab_dataset(eda)
    with t2:
        tab_protocols(read_result("protocols.json"), read_result("ablation.json"), read_result("learning_curve.json"),
                      read_result("adaptation.json"))
    with t3:
        tab_models(read_result("model_comparison.json"))
    with t4:
        tab_errors(read_result("final_reports.json"))
    with t5:
        tab_performance(read_result("latency.json"), read_result("legacy_comparison.json"))


main()
