"""Price coercion for Amazon meta rows."""

from margo.domains.fashion.loader import coerce_price, sanitize_items_frame
import pandas as pd


def test_coerce_price_placeholders():
    assert coerce_price("—") is None
    assert coerce_price("-") is None
    assert coerce_price("n/a") is None
    assert coerce_price(None) is None


def test_coerce_price_numeric_strings():
    assert coerce_price("$29.99") == 29.99
    assert coerce_price(42) == 42.0


def test_sanitize_items_frame_parquet_safe():
    df = pd.DataFrame({"parent_asin": ["A", "B"], "price": ["—", "$10.00"]})
    out = sanitize_items_frame(df)
    assert pd.isna(out.loc[0, "price"])
    assert out.loc[1, "price"] == 10.0
