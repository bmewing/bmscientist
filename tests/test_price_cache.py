from datetime import datetime, timezone

from app_discovery_agent.config import AppConfig
from app_discovery_agent.price_cache import PriceCacheDocument, PriceCacheEntry, StructuredPriceCache


def test_parse_weekly_prices_converts_eur_per_ton_to_eur_per_kg(tmp_path):
    cache = StructuredPriceCache(
        AppConfig(deepseek_api_key="x", exa_api_key="y"),
        cache_path=tmp_path / "prices.json",
    )
    html = """
    <div class="prices-history">
      <h3>Price table 2026 / 23 week</h3>
      <table class="table table-vertical-middle">
        <tr><th>Polymer type</th><th>Polymer price 2026 / 23 week</th><th>Trend</th></tr>
        <tr><td>HDPE BM</td><td>1 700 EUR / t</td><td>Difference: -166</td></tr>
        <tr><td>PVC</td><td>1 250 EUR / t</td><td>Difference: +10</td></tr>
      </table>
    </div>
    """

    entries = cache._parse_weekly_prices(html, datetime.now(timezone.utc), 1.1)

    assert len(entries) == 2
    assert entries[0].polymer_name == "HDPE BM"
    assert entries[0].price_eur_per_kg == 1.7
    assert round(entries[0].price_usd_per_kg, 3) == 1.87
    assert entries[1].price_eur_per_kg == 1.25


def test_parse_average_prices_reads_eur_per_kg_rows(tmp_path):
    cache = StructuredPriceCache(
        AppConfig(deepseek_api_key="x", exa_api_key="y"),
        cache_path=tmp_path / "prices.json",
    )
    html = """
    <table class="table">
      <tr><th>POLYMER TYPE</th><th>AVERAGE PRICE €/KG</th><th>CHANGE VS. LAST MONTH</th><th>CHANGE VS. LAST YEAR</th></tr>
      <tr><td>PP homo</td><td>2.40</td><td>4.35 % (2.3 €)</td><td>73.91 % (1.38 €)</td></tr>
      <tr><td>ABS</td><td>3,15</td><td>1 % (3.12 €)</td><td>20 % (2.62 €)</td></tr>
    </table>
    """

    entries = cache._parse_average_prices(html, datetime.now(timezone.utc), 1.05)

    assert len(entries) == 2
    assert entries[0].polymer_name == "PP homo"
    assert entries[0].price_eur_per_kg == 2.4
    assert round(entries[0].price_usd_per_kg, 2) == 2.52
    assert entries[1].price_eur_per_kg == 3.15


def test_entries_for_material_uses_alias_matching(tmp_path):
    cache = StructuredPriceCache(
        AppConfig(deepseek_api_key="x", exa_api_key="y"),
        cache_path=tmp_path / "prices.json",
    )
    document = PriceCacheDocument(
        fetched_at=datetime.now(timezone.utc),
        eur_usd_rate=1.1,
        entries=[
            PriceCacheEntry(
                source="weekly_commodity",
                page_url="https://example.com/weekly",
                label="Price table 2026 / 23 week",
                polymer_name="PS impact",
                normalized_polymer="ps impact",
                price_eur_per_kg=1.8,
                price_usd_per_kg=1.98,
                raw_price_text="1 800 EUR / t",
                fetched_at=datetime.now(timezone.utc),
            ),
            PriceCacheEntry(
                source="average_resin",
                page_url="https://example.com/avg",
                label="Average resin prices",
                polymer_name="PET",
                normalized_polymer="pet",
                price_eur_per_kg=2.1,
                price_usd_per_kg=2.31,
                raw_price_text="2.1 EUR/kg",
                fetched_at=datetime.now(timezone.utc),
            ),
        ],
    )

    hips_entries = cache.entries_for_material("High Impact Polystyrene (HIPS)", document=document)
    petg_entries = cache.entries_for_material("PETG", document=document)

    assert hips_entries[0].polymer_name == "PS impact"
    assert petg_entries[0].polymer_name == "PET"


def test_entries_for_material_does_not_match_pp_inside_gpps(tmp_path):
    cache = StructuredPriceCache(
        AppConfig(deepseek_api_key="x", exa_api_key="y"),
        cache_path=tmp_path / "prices.json",
    )
    document = PriceCacheDocument(
        fetched_at=datetime.now(timezone.utc),
        eur_usd_rate=1.1,
        entries=[
            PriceCacheEntry(
                source="average_resin",
                page_url="https://example.com/avg",
                label="Average resin prices",
                polymer_name="PP homo",
                normalized_polymer="pp homo",
                price_eur_per_kg=2.4,
                price_usd_per_kg=2.64,
                raw_price_text="2.4 EUR/kg",
                fetched_at=datetime.now(timezone.utc),
            ),
            PriceCacheEntry(
                source="average_resin",
                page_url="https://example.com/avg",
                label="Average resin prices",
                polymer_name="PS HIPS",
                normalized_polymer="ps hips",
                price_eur_per_kg=2.1,
                price_usd_per_kg=2.31,
                raw_price_text="2.1 EUR/kg",
                fetched_at=datetime.now(timezone.utc),
            ),
        ],
    )

    entries = cache.entries_for_material("General Purpose Polystyrene (GPPS)", document=document)

    assert entries[0].polymer_name == "PS HIPS"
    assert all(entry.polymer_name != "PP homo" for entry in entries)


def test_metric_for_material_converts_cached_eur_price_when_usd_is_missing(tmp_path):
    cache = StructuredPriceCache(
        AppConfig(deepseek_api_key="x", exa_api_key="y"),
        cache_path=tmp_path / "prices.json",
    )
    document = PriceCacheDocument(
        fetched_at=datetime.now(timezone.utc),
        eur_usd_rate=None,
        entries=[
            PriceCacheEntry(
                source="weekly_commodity",
                page_url="https://example.com/weekly",
                label="Price table 2026 / 23 week",
                polymer_name="PVC",
                normalized_polymer="pvc",
                price_eur_per_kg=1.25,
                price_usd_per_kg=None,
                raw_price_text="1 250 EUR / t",
                fetched_at=datetime.now(timezone.utc),
            )
        ],
    )

    metric = cache.metric_for_material("PVC", document=document)

    assert metric is not None
    assert metric.value == 1.35
    assert metric.is_inferred is True


def test_fetch_fx_rate_uses_reciprocal_of_usd_to_eur_rate(tmp_path):
    cache = StructuredPriceCache(
        AppConfig(deepseek_api_key="x", exa_api_key="y"),
        cache_path=tmp_path / "prices.json",
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"rates": {"eur": 0.8}}

    class FakeSession:
        def get(self, url, timeout):
            return FakeResponse()

    cache._session = FakeSession()

    rate, fetched_at = cache._fetch_fx_rate(existing=None)

    assert rate == 1.25
    assert fetched_at is not None


def test_fetch_fx_rate_accepts_uppercase_eur_key(tmp_path):
    cache = StructuredPriceCache(
        AppConfig(deepseek_api_key="x", exa_api_key="y"),
        cache_path=tmp_path / "prices.json",
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"rates": {"EUR": 0.8}}

    class FakeSession:
        def get(self, url, timeout):
            return FakeResponse()

    cache._session = FakeSession()

    rate, fetched_at = cache._fetch_fx_rate(existing=None)

    assert rate == 1.25
    assert fetched_at is not None
