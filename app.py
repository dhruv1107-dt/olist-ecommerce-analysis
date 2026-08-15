"""
Olist E-Commerce Dashboard
Streamlit dashboard covering revenue, retention, product/customer
concentration, and delivery performance for the Olist dataset.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Olist E-Commerce Dashboard",
    page_icon="📦",
    layout="wide",
)

FACT_PATH = "data/powerbi/fact_orders.csv"
DIM_CUSTOMERS_PATH = "data/powerbi/dim_customers.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    fact = pd.read_csv(
        FACT_PATH,
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    fact["revenue"] = fact["price"] + fact["freight_value"]

    dim_customers = pd.read_csv(DIM_CUSTOMERS_PATH, parse_dates=["first_order_date"])

    return fact, dim_customers


fact, dim_customers = load_data()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

min_date = fact["order_purchase_timestamp"].min().date()
max_date = fact["order_purchase_timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Order date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

states = sorted(fact["customer_state"].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Customer state", options=states, default=states
)

st.sidebar.caption(f"{len(fact):,} order-item rows loaded")

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
mask = (
    (fact["order_purchase_timestamp"].dt.date >= start_date)
    & (fact["order_purchase_timestamp"].dt.date <= end_date)
    & (fact["customer_state"].isin(selected_states))
)
ffact = fact.loc[mask].copy()

if ffact.empty:
    st.warning("No data matches the current filters. Adjust the date range or state selection.")
    st.stop()

# Order-level view (dedupe order-level fields that repeat across line items)
order_level = ffact.drop_duplicates(subset="order_id")[
    [
        "order_id",
        "customer_unique_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "review_score",
        "customer_state",
    ]
].copy()

customers_in_range = ffact["customer_unique_id"].unique()
fdim = dim_customers[
    dim_customers["customer_unique_id"].isin(customers_in_range)
    & dim_customers["state"].isin(selected_states)
].copy()

st.title("📦 Olist E-Commerce Dashboard")
st.caption(
    f"{start_date} → {end_date} · {len(selected_states)} state(s) selected · "
    f"{ffact['order_id'].nunique():,} orders"
)

# ---------------------------------------------------------------------------
# Section 1: Revenue Overview
# ---------------------------------------------------------------------------
st.header("1. Revenue Overview")

total_revenue = ffact["revenue"].sum()
order_count = ffact["order_id"].nunique()
aov = total_revenue / order_count if order_count else 0
avg_items_per_order = len(ffact) / order_count if order_count else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"R$ {total_revenue / 1000:,.0f}k")
c2.metric("Order Count", f"{order_count:,}")
c3.metric("Average Order Value", f"R$ {aov:,.2f}")
c4.metric("Avg Items / Order", f"{avg_items_per_order:.2f}")

monthly = (
    ffact.assign(month=ffact["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp())
    .groupby("month")
    .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
    .reset_index()
)

fig_revenue = px.line(
    monthly,
    x="month",
    y="revenue",
    markers=True,
    labels={"month": "Month", "revenue": "Revenue (R$)"},
    title="Monthly Revenue Trend",
)
fig_revenue.update_layout(hovermode="x unified")
st.plotly_chart(fig_revenue, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 2: Customer Retention
# ---------------------------------------------------------------------------
st.header("2. Customer Retention")

total_customers = fdim["customer_unique_id"].nunique()
repeat_customers = (fdim["lifetime_orders"] > 1).sum()
repeat_rate = repeat_customers / total_customers * 100 if total_customers else 0
avg_orders_per_customer = fdim["lifetime_orders"].mean() if total_customers else 0

c1, c2, c3 = st.columns(3)
c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Repeat Purchase Rate", f"{repeat_rate:.1f}%")
c3.metric("Avg Orders / Customer", f"{avg_orders_per_customer:.2f}")

order_dist = fdim["lifetime_orders"].clip(upper=5).value_counts().sort_index()
order_dist.index = order_dist.index.map(lambda x: "5+" if x == 5 else str(x))

fig_retention = px.bar(
    x=order_dist.index,
    y=order_dist.values,
    labels={"x": "Lifetime Orders per Customer", "y": "Number of Customers"},
    title="Distribution of Orders per Customer",
)
st.plotly_chart(fig_retention, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Product Concentration
# ---------------------------------------------------------------------------
st.header("3. Product Concentration")


def revenue_by_decile(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Rank entities by revenue descending and bucket into 10 equal-count deciles.
    Decile 1 = top-revenue tenth of entities."""
    grouped = df.groupby(key)["revenue"].sum().reset_index()
    grouped = grouped.sort_values("revenue", ascending=False).reset_index(drop=True)
    n = len(grouped)
    grouped["decile"] = pd.qcut(grouped.index, 10, labels=[f"D{i}" for i in range(1, 11)])
    decile_rev = grouped.groupby("decile", observed=True)["revenue"].sum().reset_index()
    decile_rev["pct_of_revenue"] = decile_rev["revenue"] / decile_rev["revenue"].sum() * 100
    return decile_rev, n


product_decile, n_products = revenue_by_decile(ffact, "product_id")
customer_decile, n_customers = revenue_by_decile(ffact, "customer_unique_id")

top_decile_product_pct = product_decile.loc[product_decile["decile"] == "D1", "pct_of_revenue"].iloc[0]
top_decile_customer_pct = customer_decile.loc[customer_decile["decile"] == "D1", "pct_of_revenue"].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Top 10% Products = Revenue Share", f"{top_decile_product_pct:.1f}%")
c2.metric("Top 10% Customers = Revenue Share", f"{top_decile_customer_pct:.1f}%")
c3.metric("Products / Customers (filtered)", f"{n_products:,} / {n_customers:,}")

col1, col2 = st.columns(2)
with col1:
    fig_prod_decile = px.bar(
        product_decile,
        x="decile",
        y="pct_of_revenue",
        labels={"decile": "Product Decile (D1 = top)", "pct_of_revenue": "% of Total Revenue"},
        title="Revenue by Product Decile",
    )
    st.plotly_chart(fig_prod_decile, use_container_width=True)
with col2:
    fig_cust_decile = px.bar(
        customer_decile,
        x="decile",
        y="pct_of_revenue",
        labels={"decile": "Customer Decile (D1 = top)", "pct_of_revenue": "% of Total Revenue"},
        title="Revenue by Customer Decile",
    )
    st.plotly_chart(fig_cust_decile, use_container_width=True)

top_categories = (
    ffact.dropna(subset=["product_category_name"])
    .groupby("product_category_name")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

fig_categories = px.bar(
    x=top_categories.values,
    y=top_categories.index,
    orientation="h",
    labels={"x": "Revenue (R$)", "y": "Product Category"},
    title="Top 10 Categories by Revenue",
)
st.plotly_chart(fig_categories, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Delivery Performance
# ---------------------------------------------------------------------------
st.header("4. Delivery Performance")

delivered = order_level.dropna(
    subset=["order_delivered_customer_date", "order_estimated_delivery_date", "review_score"]
).copy()
delivered["days_early"] = (
    delivered["order_estimated_delivery_date"] - delivered["order_delivered_customer_date"]
).dt.days  # positive = delivered early, negative = delivered late

avg_review_score = delivered["review_score"].mean() if not delivered.empty else 0
pct_late = (delivered["days_early"] < 0).mean() * 100 if not delivered.empty else 0
avg_days_early = delivered["days_early"].mean() if not delivered.empty else 0

c1, c2, c3 = st.columns(3)
c1.metric("Avg Review Score", f"{avg_review_score:.2f} / 5")
c2.metric("% Orders Delivered Late", f"{pct_late:.1f}%")
c3.metric("Avg Days Early / Late", f"{avg_days_early:+.1f} days")


def bucket_days(d):
    if d < -14:
        return "Late 15+ days"
    if d < -7:
        return "Late 8-14 days"
    if d < 0:
        return "Late 1-7 days"
    if d == 0:
        return "On time"
    if d <= 7:
        return "Early 1-7 days"
    if d <= 14:
        return "Early 8-14 days"
    return "Early 15+ days"


bucket_order = [
    "Late 15+ days",
    "Late 8-14 days",
    "Late 1-7 days",
    "On time",
    "Early 1-7 days",
    "Early 8-14 days",
    "Early 15+ days",
]

delivered["bucket"] = delivered["days_early"].apply(bucket_days)
bucket_stats = (
    delivered.groupby("bucket")["review_score"]
    .agg(avg_review_score="mean", orders="count")
    .reindex(bucket_order)
    .dropna()
    .reset_index()
)

fig_delivery = go.Figure()
fig_delivery.add_bar(
    x=bucket_stats["bucket"],
    y=bucket_stats["avg_review_score"],
    text=bucket_stats["orders"].apply(lambda x: f"n={x:,}"),
    textposition="outside",
    marker_color="#2E86AB",
    name="Avg review score",
)
fig_delivery.update_layout(
    title="Average Review Score by Delivery Timing",
    xaxis_title="Delivery Timing (vs. Estimate)",
    yaxis_title="Average Review Score",
    yaxis_range=[0, 5.5],
)
st.plotly_chart(fig_delivery, use_container_width=True)

st.caption(
    "Delivery timing = estimated delivery date minus actual delivered date. "
    "Positive means delivered early; negative means delivered late. "
    "Review score and delivery dates are deduplicated to one row per order."
)
