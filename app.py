"""
app.py — Streamlit Dashboard for Factory Reallocation & Shipping Optimization.

A professional, interactive analytics dashboard that lets decision-makers
explore factory and distribution center data, run LP-based shipping
optimization, compare against baselines, and export results.

NOTE: All sample data is illustrative and does not represent actual
Nassau Candy business data.

Launch:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import os

from data_loader import (
    load_all_data,
    validate_factories,
    validate_destinations,
    validate_shipping_costs,
    validate_cross_references,
    build_cost_matrix,
    DataValidationError,
)
from optimizer import (
    optimize_shipping,
    build_baseline_allocation,
    compute_savings,
    OptimizationResult,
)
from recommendations import (
    generate_recommendations,
    format_recommendations_text,
    build_summary_report,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy — Shipping Optimization",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for professional styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- global ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---- KPI cards ---- */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #f1f5f9;
}
.kpi-value.green  { color: #4ade80; }
.kpi-value.blue   { color: #60a5fa; }
.kpi-value.amber  { color: #fbbf24; }
.kpi-value.red    { color: #f87171; }

/* ---- section headers ---- */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    padding-bottom: 8px;
    border-bottom: 2px solid #3b82f6;
}

/* ---- sidebar branding ---- */
.sidebar-brand {
    font-size: 1.1rem;
    font-weight: 700;
    color: #60a5fa;
    text-align: center;
    padding: 4px 0 10px 0;
    border-bottom: 1px solid #334155;
    margin-bottom: 12px;
}

/* ---- disclaimer banner ---- */
.disclaimer {
    background: #1e1b4b;
    border-left: 4px solid #818cf8;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #c7d2fe;
    margin-bottom: 16px;
}

/* ---- recommendation card ---- */
.rec-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left: 4px solid #3b82f6;
    font-size: 0.92rem;
    line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🏭 Nassau Candy<br>Shipping Optimizer</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Overview", "📁 Data Management", "⚙️ Optimization",
         "📈 Visual Analytics", "💡 Recommendations", "📥 Export Results"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("##### ⚙️ Settings")
    include_prod_cost = st.checkbox(
        "Include production costs",
        value=False,
        help="When enabled the model minimises production + shipping cost.",
    )

    st.markdown("---")
    st.caption("© 2026 — Illustrative Sample Data\nNot actual Nassau Candy data.")

# ---------------------------------------------------------------------------
# Data loading (session state)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _default_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def load_default_data():
    """Load the shipped sample data into session state."""
    try:
        f, d, s = load_all_data(
            _default_path("factories.csv"),
            _default_path("destinations.csv"),
            _default_path("shipping_costs.csv"),
        )
        st.session_state["factories"] = f
        st.session_state["destinations"] = d
        st.session_state["shipping_costs"] = s
        st.session_state["data_source"] = "sample"
        st.session_state["data_error"] = None
    except Exception as e:
        st.session_state["data_error"] = str(e)


if "factories" not in st.session_state:
    load_default_data()


def _get_data():
    return (
        st.session_state.get("factories"),
        st.session_state.get("destinations"),
        st.session_state.get("shipping_costs"),
    )

# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def kpi(label: str, value_str: str, color: str = ""):
    cls = f" {color}" if color else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value{cls}">{value_str}</div>
    </div>"""


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def disclaimer():
    st.markdown(
        '<div class="disclaimer">'
        '📌 <strong>Disclaimer:</strong> All data and results shown are '
        '<em>illustrative sample data</em> created for demonstration purposes. '
        'They do not represent actual Nassau Candy business data or verified savings.'
        '</div>',
        unsafe_allow_html=True,
    )


# ===================================================================
# PAGE: Overview
# ===================================================================
if page == "📊 Overview":
    st.title("📊 Supply-Chain Optimization Dashboard")
    disclaimer()

    factories, destinations, shipping_costs = _get_data()
    if st.session_state.get("data_error"):
        st.error(f"Data loading error: {st.session_state['data_error']}")
        st.stop()

    total_cap = factories["production_capacity"].sum()
    total_dem = destinations["demand"].sum()

    # Run optimized + baseline once for overview metrics
    optimized = optimize_shipping(factories, destinations, shipping_costs, include_prod_cost)
    baseline = build_baseline_allocation(factories, destinations, shipping_costs, include_prod_cost)
    st.session_state["optimized"] = optimized
    st.session_state["baseline"] = baseline

    # Row 1 — basic KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Total Factories", str(len(factories)), "blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Distribution Centers", str(len(destinations)), "blue"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Production Capacity", f"{total_cap:,.0f}", "amber"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Total Demand", f"{total_dem:,.0f}", "amber"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — cost KPIs
    if optimized.status == "Optimal" and baseline.status == "Optimal":
        savings = compute_savings(baseline.total_cost, optimized.total_cost)
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown(kpi("Baseline Cost", f"${baseline.total_cost:,.2f}"), unsafe_allow_html=True)
        with c6:
            st.markdown(kpi("Optimized Cost", f"${optimized.total_cost:,.2f}", "green"), unsafe_allow_html=True)
        with c7:
            st.markdown(kpi("Cost Savings", f"${savings['cost_savings']:,.2f}", "green"), unsafe_allow_html=True)
        with c8:
            st.markdown(kpi("Savings %", f"{savings['savings_pct']:.1f}%", "green"), unsafe_allow_html=True)
    elif optimized.status != "Optimal":
        st.warning(f"Optimization infeasible: {optimized.infeasibility_reason}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Capacity vs Demand quick chart
    section("Capacity vs Demand Snapshot")
    cap_df = factories[["factory_name", "production_capacity"]].copy()
    fig_cap = px.bar(
        cap_df, x="factory_name", y="production_capacity",
        color_discrete_sequence=["#3b82f6"],
        labels={"factory_name": "Factory", "production_capacity": "Units"},
        title="Factory Production Capacities",
    )
    fig_cap.add_hline(
        y=total_dem / len(factories),
        line_dash="dot", line_color="#fbbf24",
        annotation_text="Avg demand per DC",
    )
    fig_cap.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        title_font_size=14,
    )
    st.plotly_chart(fig_cap, use_container_width=True)


# ===================================================================
# PAGE: Data Management
# ===================================================================
elif page == "📁 Data Management":
    st.title("📁 Data Management")
    disclaimer()

    tab_view, tab_upload = st.tabs(["📋 View Current Data", "📤 Upload CSV"])

    # ---- View tab ----
    with tab_view:
        factories, destinations, shipping_costs = _get_data()
        if factories is not None:
            section("Factory Data")
            st.dataframe(factories, use_container_width=True, hide_index=True)

            section("Distribution Center Demand")
            st.dataframe(destinations, use_container_width=True, hide_index=True)

            section("Shipping Cost Matrix")
            cost_matrix = build_cost_matrix(factories, destinations, shipping_costs)
            st.dataframe(cost_matrix, use_container_width=True)

            section("Raw Shipping Costs")
            st.dataframe(shipping_costs, use_container_width=True, hide_index=True)
        else:
            st.info("No data loaded. Upload CSV files or check the data/ folder.")

    # ---- Upload tab ----
    with tab_upload:
        st.markdown(
            "Upload your own CSV files. The system will validate required columns "
            "and data integrity before accepting them."
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            f_file = st.file_uploader("Factories CSV", type="csv", key="up_f")
        with col_b:
            d_file = st.file_uploader("Destinations CSV", type="csv", key="up_d")
        with col_c:
            s_file = st.file_uploader("Shipping Costs CSV", type="csv", key="up_s")

        if st.button("✅ Validate & Load", type="primary"):
            try:
                new_f = pd.read_csv(f_file) if f_file else st.session_state["factories"]
                new_d = pd.read_csv(d_file) if d_file else st.session_state["destinations"]
                new_s = pd.read_csv(s_file) if s_file else st.session_state["shipping_costs"]

                validate_factories(new_f)
                validate_destinations(new_d)
                validate_shipping_costs(new_s)
                validate_cross_references(new_f, new_d, new_s)

                st.session_state["factories"] = new_f
                st.session_state["destinations"] = new_d
                st.session_state["shipping_costs"] = new_s
                st.session_state["data_source"] = "uploaded"
                st.session_state["data_error"] = None
                # Clear cached optimisation
                st.session_state.pop("optimized", None)
                st.session_state.pop("baseline", None)
                st.success("✅ Data validated and loaded successfully!")
            except DataValidationError as e:
                st.error(f"❌ Validation Error: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {e}")

        st.markdown("---")
        st.markdown("##### Required Column Formats")
        st.markdown("""
| File | Required Columns |
|---|---|
| `factories.csv` | `factory_id`, `factory_name`, `location`, `production_capacity` |
| `destinations.csv` | `destination_id`, `destination_name`, `location`, `demand` |
| `shipping_costs.csv` | `factory_id`, `destination_id`, `shipping_cost_per_unit` |
        """)


# ===================================================================
# PAGE: Optimization
# ===================================================================
elif page == "⚙️ Optimization":
    st.title("⚙️ Shipping Optimization")
    disclaimer()

    factories, destinations, shipping_costs = _get_data()

    st.markdown(
        "Click **Optimize Shipping** to compute the minimum-cost allocation "
        "using linear programming. The model finds the shipment quantities "
        "from each factory to each distribution center that satisfy all "
        "demand while respecting capacity constraints."
    )

    if st.button("🚀 Optimize Shipping", type="primary", use_container_width=True):
        with st.spinner("Running LP solver…"):
            optimized = optimize_shipping(
                factories, destinations, shipping_costs, include_prod_cost
            )
            baseline = build_baseline_allocation(
                factories, destinations, shipping_costs, include_prod_cost
            )
            st.session_state["optimized"] = optimized
            st.session_state["baseline"] = baseline

    optimized: OptimizationResult = st.session_state.get("optimized")
    baseline: OptimizationResult = st.session_state.get("baseline")

    if optimized is None:
        st.info("Press the button above to run the optimiser.")
        st.stop()

    if optimized.status != "Optimal":
        st.error(f"⚠️ **{optimized.status}** — {optimized.infeasibility_reason}")
        st.stop()

    st.success(f"✅ Optimal solution found — Total cost: **${optimized.total_cost:,.2f}**")

    # ---- Shipment plan ----
    section("Recommended Shipment Plan")
    st.dataframe(
        optimized.shipment_plan.style.format({
            "quantity_shipped": "{:,.0f}",
            "shipping_cost_per_unit": "${:.2f}",
            "production_cost_per_unit": "${:.2f}",
            "total_route_cost": "${:,.2f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # ---- Factory utilization ----
    section("Factory Capacity Utilization")
    st.dataframe(
        optimized.factory_utilization.style.format({
            "production_capacity": "{:,.0f}",
            "allocated_shipments": "{:,.0f}",
            "remaining_capacity": "{:,.0f}",
            "utilization_pct": "{:.1f}%",
        }).background_gradient(
            subset=["utilization_pct"], cmap="RdYlGn", vmin=0, vmax=100
        ),
        use_container_width=True, hide_index=True,
    )

    # ---- Recommended factory per DC ----
    section("Recommended Primary Factory per Distribution Center")
    if not optimized.shipment_plan.empty:
        primary_map = (
            optimized.shipment_plan
            .sort_values("quantity_shipped", ascending=False)
            .drop_duplicates(subset="destination_id", keep="first")
            [["destination_id", "destination_name", "factory_id", "factory_name",
              "quantity_shipped", "shipping_cost_per_unit"]]
        )
        st.dataframe(primary_map, use_container_width=True, hide_index=True)


# ===================================================================
# PAGE: Visual Analytics
# ===================================================================
elif page == "📈 Visual Analytics":
    st.title("📈 Visual Analytics")
    disclaimer()

    factories, destinations, shipping_costs = _get_data()
    optimized: OptimizationResult = st.session_state.get("optimized")
    baseline: OptimizationResult = st.session_state.get("baseline")

    if optimized is None or optimized.status != "Optimal":
        st.info("Run the optimisation first (⚙️ Optimization page) to see charts.")
        st.stop()

    plan = optimized.shipment_plan
    util = optimized.factory_utilization
    layout_defaults = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        title_font_size=14,
    )

    # ---- 1. Capacity vs Allocated ----
    section("Factory Capacity vs Allocated Shipments")
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=util["factory_name"], y=util["production_capacity"],
        name="Capacity", marker_color="#3b82f6",
    ))
    fig1.add_trace(go.Bar(
        x=util["factory_name"], y=util["allocated_shipments"],
        name="Allocated", marker_color="#22c55e",
    ))
    fig1.update_layout(
        barmode="group", **layout_defaults,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ---- 2. Shipping cost by DC ----
    section("Total Shipping Cost by Distribution Center")
    cost_by_dc = (
        plan.groupby("destination_name")["total_route_cost"]
        .sum().reset_index()
        .sort_values("total_route_cost", ascending=True)
    )
    fig2 = px.bar(
        cost_by_dc, y="destination_name", x="total_route_cost",
        orientation="h",
        color_discrete_sequence=["#f59e0b"],
        labels={"destination_name": "Distribution Center", "total_route_cost": "Total Cost ($)"},
    )
    fig2.update_layout(**layout_defaults)
    st.plotly_chart(fig2, use_container_width=True)

    # ---- 3. Shipment quantities by factory ----
    section("Shipment Quantities by Factory")
    qty_by_factory = (
        plan.groupby(["factory_name", "destination_name"])["quantity_shipped"]
        .sum().reset_index()
    )
    fig3 = px.bar(
        qty_by_factory, x="factory_name", y="quantity_shipped",
        color="destination_name",
        barmode="stack",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"factory_name": "Factory", "quantity_shipped": "Units Shipped",
                "destination_name": "Destination"},
    )
    fig3.update_layout(**layout_defaults)
    st.plotly_chart(fig3, use_container_width=True)

    # ---- 4. Baseline vs Optimized ----
    if baseline and baseline.status == "Optimal":
        section("Baseline vs Optimized Cost Comparison")
        comp_df = pd.DataFrame({
            "Allocation": ["Baseline (Heuristic)", "Optimized (LP)"],
            "Total Cost": [baseline.total_cost, optimized.total_cost],
        })
        fig4 = px.bar(
            comp_df, x="Allocation", y="Total Cost",
            color="Allocation",
            color_discrete_map={
                "Baseline (Heuristic)": "#ef4444",
                "Optimized (LP)": "#22c55e",
            },
            text="Total Cost",
        )
        fig4.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
        fig4.update_layout(showlegend=False, **layout_defaults)
        st.plotly_chart(fig4, use_container_width=True)

    # ---- 5. Network visualization (Sankey) ----
    section("Factory → Distribution Center Network")
    factory_names_list = plan["factory_name"].unique().tolist()
    dest_names_list = plan["destination_name"].unique().tolist()
    all_labels = factory_names_list + dest_names_list

    source_idx = [factory_names_list.index(r) for r in plan["factory_name"]]
    target_idx = [len(factory_names_list) + dest_names_list.index(r) for r in plan["destination_name"]]

    factory_colors = ["#3b82f6", "#6366f1", "#8b5cf6"]
    dest_colors = ["#f59e0b", "#ef4444", "#22c55e", "#06b6d4"]
    node_colors = factory_colors[:len(factory_names_list)] + dest_colors[:len(dest_names_list)]

    fig5 = go.Figure(go.Sankey(
        node=dict(
            label=all_labels,
            color=node_colors,
            pad=20,
            thickness=24,
        ),
        link=dict(
            source=source_idx,
            target=target_idx,
            value=plan["quantity_shipped"].tolist(),
            color=["rgba(59,130,246,0.3)"] * len(plan),
        ),
    ))
    fig5.update_layout(
        title_text="Shipment Flow Network",
        **layout_defaults,
        height=420,
    )
    st.plotly_chart(fig5, use_container_width=True)


# ===================================================================
# PAGE: Recommendations
# ===================================================================
elif page == "💡 Recommendations":
    st.title("💡 Business Recommendations")
    disclaimer()

    factories, destinations, shipping_costs = _get_data()
    optimized = st.session_state.get("optimized")
    baseline = st.session_state.get("baseline")

    if optimized is None or optimized.status != "Optimal":
        st.info("Run the optimisation first (⚙️ Optimization page) to generate recommendations.")
        st.stop()

    recs = generate_recommendations(
        optimized, baseline, factories, destinations, shipping_costs
    )
    st.session_state["recommendations"] = recs

    if not recs:
        st.info("No recommendations generated — check optimisation output.")
    else:
        for rec in recs:
            st.markdown(
                f'<div class="rec-card">{rec}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption(
        "These recommendations are generated dynamically from the currently "
        "loaded data and optimization results. They are illustrative and "
        "should be validated against actual business conditions."
    )


# ===================================================================
# PAGE: Export Results
# ===================================================================
elif page == "📥 Export Results":
    st.title("📥 Export Results")
    disclaimer()

    factories, destinations, shipping_costs = _get_data()
    optimized: OptimizationResult = st.session_state.get("optimized")
    baseline: OptimizationResult = st.session_state.get("baseline")

    if optimized is None or optimized.status != "Optimal":
        st.info("Run the optimisation first to enable downloads.")
        st.stop()

    section("Download Optimized Shipment Plan")
    plan_csv = optimized.shipment_plan.to_csv(index=False)
    st.download_button(
        "⬇️ Download Shipment Plan (CSV)",
        data=plan_csv,
        file_name="optimized_shipment_plan.csv",
        mime="text/csv",
    )

    section("Download Cost Comparison Report")
    if baseline and baseline.status == "Optimal":
        savings = compute_savings(baseline.total_cost, optimized.total_cost)
        comp_df = pd.DataFrame([{
            "Metric": "Baseline Cost",
            "Value": f"${savings['baseline_cost']:,.2f}",
        }, {
            "Metric": "Optimized Cost",
            "Value": f"${savings['optimized_cost']:,.2f}",
        }, {
            "Metric": "Cost Savings",
            "Value": f"${savings['cost_savings']:,.2f}",
        }, {
            "Metric": "Savings %",
            "Value": f"{savings['savings_pct']:.1f}%",
        }])
        comp_csv = comp_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Cost Comparison (CSV)",
            data=comp_csv,
            file_name="cost_comparison.csv",
            mime="text/csv",
        )
    else:
        st.info("Baseline not available for comparison.")

    section("Download Summary Report")
    recs = st.session_state.get("recommendations")
    if recs is None:
        recs = generate_recommendations(
            optimized, baseline, factories, destinations, shipping_costs
        )
    report_text = build_summary_report(
        optimized, baseline, factories, destinations, shipping_costs, recs,
    )
    st.download_button(
        "⬇️ Download Summary Report (TXT)",
        data=report_text,
        file_name="optimization_summary_report.txt",
        mime="text/plain",
    )

    st.markdown("---")
    st.caption("All exported files contain illustrative sample results.")
