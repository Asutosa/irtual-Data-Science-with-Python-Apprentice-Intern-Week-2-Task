
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output_week2"
CHART_DIR = OUT_DIR / "charts"
OUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

WDI_FILE = BASE_DIR / "WDICSV.csv"
COUNTRY_FILE = BASE_DIR / "WDICountry.csv"
SERIES_FILE = BASE_DIR / "WDISeries.csv"

# Carefully selected indicators for a connected story about development.
INDICATORS = {
    "NY.GDP.PCAP.KD": "GDP per capita (constant 2015 US$)",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
    "SP.POP.TOTL": "Population, total",
    "SL.UEM.TOTL.ZS": "Unemployment, total (% of total labor force) (modeled ILO estimate)",
    "SH.DYN.MORT": "Mortality rate, under-5 (per 1,000 live births)",
    "EG.ELC.ACCS.ZS": "Access to electricity (% of population)",
}
START_YEAR = 2000
END_YEAR = 2024
FOCUS_COUNTRIES = ["India", "China", "United States", "Brazil", "Indonesia", "South Africa",
                   "Germany", "Japan", "Nigeria", "Bangladesh"]

def require_files():
    missing = [str(p.name) for p in [WDI_FILE, COUNTRY_FILE, SERIES_FILE] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s): " + ", ".join(missing) +
            "\nPlace this script beside WDICSV.csv, WDICountry.csv and WDISeries.csv."
        )

def add_chart_caption(doc, title, explanation, why):
    doc.add_heading(title, level=2)
    doc.add_paragraph(explanation)
    p = doc.add_paragraph()
    r = p.add_run("Why this chart: ")
    r.bold = True
    p.add_run(why)

def savefig(name):
    path = CHART_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path

def main():
    require_files()
    print("Reading WDI files. WDICSV.csv is large, so this may take a minute...")

    # Read only selected indicator rows from the wide WDI file.
    # This avoids loading the full dataset into memory after parsing.
    raw = pd.read_csv(WDI_FILE, low_memory=False)
    year_cols = [c for c in raw.columns if str(c).isdigit() and START_YEAR <= int(c) <= END_YEAR]
    needed_codes = list(INDICATORS.keys())
    data = raw.loc[raw["Indicator Code"].isin(needed_codes),
                   ["Country Name", "Country Code", "Indicator Name", "Indicator Code"] + year_cols].copy()
    del raw

    if data.empty:
        raise ValueError("No selected indicator codes were found in WDICSV.csv. Check the dataset version.")

    # Convert wide format to long format for time-series and comparisons.
    long = data.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        value_vars=year_cols, var_name="Year", value_name="Value"
    )
    long["Year"] = pd.to_numeric(long["Year"], errors="coerce")
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
    long = long.dropna(subset=["Year"])
    long["Indicator"] = long["Indicator Code"].map(INDICATORS)

    countries = pd.read_csv(COUNTRY_FILE, low_memory=False)
    country_cols = ["Country Code", "Short Name", "Region", "Income Group"]
    countries = countries[[c for c in country_cols if c in countries.columns]].drop_duplicates("Country Code")
    long = long.merge(countries, on="Country Code", how="left")

    # WDI contains regional/income aggregates as well as individual economies.
    # Keep only economies with a named region for country-level comparisons.
    long["Region"] = long["Region"].fillna("Unclassified")
    economy = long[long["Region"].ne("Aggregates")].copy()
    economy = economy[economy["Country Code"].notna()]

    # Save analysis-ready long-format data.
    economy.to_csv(OUT_DIR / "wdi_selected_indicators_long.csv", index=False)

    # Data quality and coverage summary.
    quality = pd.DataFrame({
        "Indicator Code": needed_codes,
        "Indicator": [INDICATORS[c] for c in needed_codes],
        "Rows available": [int(data.loc[data["Indicator Code"] == c, year_cols].notna().sum().sum()) for c in needed_codes],
        "Countries with any data": [int(data.loc[data["Indicator Code"] == c, "Country Code"].nunique()) for c in needed_codes],
    })
    quality.to_csv(OUT_DIR / "indicator_coverage.csv", index=False)

    # Latest year with useful coverage for each indicator (at least 30 economies).
    coverage = economy.dropna(subset=["Value"]).groupby(["Indicator Code", "Year"])["Country Code"].nunique().reset_index(name="n")
    latest_by_code = {}
    for code in needed_codes:
        eligible = coverage[(coverage["Indicator Code"] == code) & (coverage["n"] >= 30)]
        if not eligible.empty:
            latest_by_code[code] = int(eligible["Year"].max())

    # Most indicators should have a common comparison year; choose the latest year
    # where GDP, life expectancy and internet coverage all have sufficient coverage.
    core = ["NY.GDP.PCAP.KD", "SP.DYN.LE00.IN", "IT.NET.USER.ZS"]
    common_years = None
    for code in core:
        years = set(coverage[(coverage["Indicator Code"] == code) & (coverage["n"] >= 50)]["Year"].astype(int))
        common_years = years if common_years is None else common_years.intersection(years)
    comparison_year = max(common_years) if common_years else min(END_YEAR, int(economy["Year"].max()))
    print("Country comparison year:", comparison_year)

    def indicator_frame(code):
        return economy[economy["Indicator Code"].eq(code)].copy()

    # Chart 1: Global development trends indexed to 2000 = 100.
    trend_codes = ["NY.GDP.PCAP.KD", "SP.DYN.LE00.IN", "IT.NET.USER.ZS"]
    trend_labels = {
        "NY.GDP.PCAP.KD": "GDP per capita",
        "SP.DYN.LE00.IN": "Life expectancy",
        "IT.NET.USER.ZS": "Internet use",
    }
    trend = economy[economy["Indicator Code"].isin(trend_codes)].groupby(
        ["Indicator Code", "Year"], as_index=False
    )["Value"].median()
    base = trend[trend["Year"].eq(2000)][["Indicator Code", "Value"]].rename(columns={"Value": "Base"})
    trend = trend.merge(base, on="Indicator Code", how="left")
    trend["Index (2000=100)"] = 100 * trend["Value"] / trend["Base"]
    plt.figure(figsize=(10, 5.8))
    for code in trend_codes:
        z = trend[trend["Indicator Code"].eq(code)].dropna(subset=["Index (2000=100)"])
        if not z.empty:
            plt.plot(z["Year"], z["Index (2000=100)"], marker="o", markersize=2.5,
                     linewidth=2.2, label=trend_labels[code])
    plt.axhline(100, color="gray", linestyle="--", linewidth=1)
    plt.title("How have key development indicators changed since 2000?")
    plt.xlabel("Year")
    plt.ylabel("Index (2000 = 100)")
    plt.legend()
    c1 = savefig("01_global_development_trends.png")

    # Chart 2: India's long-run development dashboard (separate panels because units differ).
    india = economy[economy["Country Name"].eq("India") & economy["Indicator Code"].isin(trend_codes)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, code in zip(axes, trend_codes):
        z = india[india["Indicator Code"].eq(code)].dropna(subset=["Value"])
        ax.plot(z["Year"], z["Value"], marker="o", markersize=2.5, linewidth=2, label=INDICATORS[code])
        ax.set_ylabel({ "NY.GDP.PCAP.KD": "2015 US$",
                        "SP.DYN.LE00.IN": "Years",
                        "IT.NET.USER.ZS": "% of population"}[code])
        ax.set_title(INDICATORS[code], loc="left", fontsize=10)
    axes[-1].set_xlabel("Year")
    fig.suptitle("India's development trajectory", fontsize=15, y=1.01)
    c2 = savefig("02_india_development_trajectory.png")

    # Chart 3: GDP per capita vs life expectancy, colored by internet use.
    latest = economy[economy["Year"].eq(comparison_year)]
    wide = latest[latest["Indicator Code"].isin(core)].pivot_table(
        index=["Country Code", "Country Name", "Region", "Income Group"],
        columns="Indicator Code", values="Value", aggfunc="first"
    ).reset_index()
    wide = wide.rename(columns={
        "NY.GDP.PCAP.KD": "GDP per capita",
        "SP.DYN.LE00.IN": "Life expectancy",
        "IT.NET.USER.ZS": "Internet use (%)",
    }).dropna(subset=["GDP per capita", "Life expectancy", "Internet use (%)"])
    wide = wide[(wide["GDP per capita"] > 0) & (wide["Life expectancy"].between(30, 90))]
    plt.figure(figsize=(10, 6.5))
    sns.scatterplot(data=wide, x="GDP per capita", y="Life expectancy",
                    hue="Internet use (%)", size="Internet use (%)",
                    sizes=(25, 220), palette="viridis", alpha=.8, legend="brief")
    plt.xscale("log")
    plt.title(f"Do wealthier economies tend to have longer life expectancy? ({comparison_year})")
    plt.xlabel("GDP per capita (constant 2015 US$, logarithmic scale)")
    plt.ylabel("Life expectancy at birth (years)")
    c3 = savefig("03_gdp_life_expectancy_scatter.png")

    # Annotate a handful of selected countries on the scatter for readability.
    plt.figure(figsize=(10, 6.5))
    sns.scatterplot(data=wide, x="GDP per capita", y="Life expectancy",
                    hue="Income Group", alpha=.75, s=65)
    plt.xscale("log")
    for _, row in wide[wide["Country Name"].isin(["India", "China", "United States", "Brazil", "Nigeria", "Japan"])].iterrows():
        plt.annotate(row["Country Name"], (row["GDP per capita"], row["Life expectancy"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.title(f"Country comparison: income, prosperity and longevity ({comparison_year})")
    plt.xlabel("GDP per capita (constant 2015 US$, logarithmic scale)")
    plt.ylabel("Life expectancy at birth (years)")
    c4 = savefig("04_country_comparison_annotated.png")

    # Chart 5: regional median internet access.
    internet = economy[(economy["Indicator Code"] == "IT.NET.USER.ZS") & (economy["Year"] == comparison_year)]
    region = internet.dropna(subset=["Value"]).groupby("Region", as_index=False)["Value"].median()
    region = region[~region["Region"].isin(["Unclassified", "Aggregates"])].sort_values("Value")
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=region, y="Region", x="Value", hue="Region", legend=False)
    plt.title(f"Digital divide: median internet use by region ({comparison_year})")
    plt.xlabel("Individuals using the Internet (% of population)")
    plt.ylabel("")
    plt.xlim(0, 100)
    c5 = savefig("05_regional_internet_access.png")

    # Chart 6: under-5 mortality trend, selected countries.
    mortality_codes = ["SH.DYN.MORT"]
    selected = economy[
        economy["Indicator Code"].isin(mortality_codes) &
        economy["Country Name"].isin(FOCUS_COUNTRIES) &
        economy["Year"].between(2000, END_YEAR)
    ]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=selected, x="Year", y="Value", hue="Country Name", linewidth=2)
    plt.title("Child health progress: under-five mortality over time")
    plt.xlabel("Year")
    plt.ylabel("Under-five deaths per 1,000 live births")
    plt.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left")
    c6 = savefig("06_under5_mortality_trends.png")

    # Chart 7: electricity access and internet use.
    access_codes = ["EG.ELC.ACCS.ZS", "IT.NET.USER.ZS"]
    access = economy[(economy["Indicator Code"].isin(access_codes)) & (economy["Year"] == comparison_year)]
    access_wide = access.pivot_table(index=["Country Code", "Country Name", "Region"],
                                     columns="Indicator Code", values="Value", aggfunc="first").reset_index()
    access_wide = access_wide.rename(columns={
        "EG.ELC.ACCS.ZS": "Electricity access (%)",
        "IT.NET.USER.ZS": "Internet use (%)"
    }).dropna(subset=["Electricity access (%)", "Internet use (%)"])
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=access_wide, x="Electricity access (%)", y="Internet use (%)",
                    hue="Region", alpha=.8, s=65)
    plt.title(f"Digital connectivity and electricity access ({comparison_year})")
    plt.xlabel("Access to electricity (% of population)")
    plt.ylabel("Internet users (% of population)")
    c7 = savefig("07_electricity_vs_internet.png")

    # Chart 8: correlation heatmap using one row per economy in comparison year.
    corr_codes = ["NY.GDP.PCAP.KD", "SP.DYN.LE00.IN", "IT.NET.USER.ZS",
                  "SL.UEM.TOTL.ZS", "SH.DYN.MORT", "EG.ELC.ACCS.ZS"]
    corr_long = latest[latest["Indicator Code"].isin(corr_codes)]
    corr_wide = corr_long.pivot_table(index=["Country Code", "Country Name"],
                                      columns="Indicator Code", values="Value", aggfunc="first")
    corr_wide = corr_wide.rename(columns={
        "NY.GDP.PCAP.KD": "GDP per capita",
        "SP.DYN.LE00.IN": "Life expectancy",
        "IT.NET.USER.ZS": "Internet use",
        "SL.UEM.TOTL.ZS": "Unemployment",
        "SH.DYN.MORT": "Under-five mortality",
        "EG.ELC.ACCS.ZS": "Electricity access",
    })
    corr = corr_wide.corr(min_periods=20)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1)
    plt.title(f"How do development indicators move together? ({comparison_year})")
    c8 = savefig("08_development_correlation_heatmap.png")

    # Summary numbers for reproducible narrative.
    def country_value(code, country, year):
        s = economy[(economy["Indicator Code"] == code) &
                    (economy["Country Name"] == country) &
                    (economy["Year"] == year)]["Value"].dropna()
        return float(s.iloc[0]) if len(s) else np.nan

    india_gdp_2000 = country_value("NY.GDP.PCAP.KD", "India", 2000)
    india_gdp_end = country_value("NY.GDP.PCAP.KD", "India", END_YEAR)
    india_life_2000 = country_value("SP.DYN.LE00.IN", "India", 2000)
    india_life_end = country_value("SP.DYN.LE00.IN", "India", END_YEAR)
    india_net_2000 = country_value("IT.NET.USER.ZS", "India", 2000)
    india_net_end = country_value("IT.NET.USER.ZS", "India", END_YEAR)
    india_mort_2000 = country_value("SH.DYN.MORT", "India", 2000)
    india_mort_end = country_value("SH.DYN.MORT", "India", END_YEAR)

    def change_text(a, b, unit=""):
        if pd.isna(a) or pd.isna(b):
            return "Not enough data for the full period in this release."
        return f"{a:,.2f}{unit} in 2000 to {b:,.2f}{unit} in {END_YEAR}"

    gdp_life_corr = wide["GDP per capita"].corr(wide["Life expectancy"]) if len(wide) > 2 else np.nan
    regional_top = region.sort_values("Value", ascending=False).head(1)
    regional_low = region.sort_values("Value").head(1)
    top_region_text = f"{regional_top.iloc[0]['Region']} ({regional_top.iloc[0]['Value']:.1f}%)" if not regional_top.empty else "not available"
    low_region_text = f"{regional_low.iloc[0]['Region']} ({regional_low.iloc[0]['Value']:.1f}%)" if not regional_low.empty else "not available"

    # Save selected comparison tables.
    wide.to_csv(OUT_DIR / "country_comparison_latest_year.csv", index=False)
    region.to_csv(OUT_DIR / "regional_internet_medians.csv", index=False)
    corr.to_csv(OUT_DIR / "indicator_correlations.csv")

    print("\nSUCCESS — Analysis complete.")
    print("Charts:", CHART_DIR)
    print("Data exports:", OUT_DIR)
    print("Comparison year:", comparison_year)

if __name__ == "__main__":
    main()
